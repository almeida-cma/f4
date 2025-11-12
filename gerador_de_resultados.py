# gerador_de_resultados.py (v4 - Correção Definitiva do KeyError)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os
import warnings

# Ignora avisos que não são críticos para esta implementação
warnings.filterwarnings("ignore")

# Importações da biblioteca de otimização Pymoo
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter

# --- 0. CONFIGURAÇÕES GLOBAIS ---
RESULTS_DIR = 'resultados_simulacao'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- INÍCIO DO PROCESSO ---
print("🚀 INICIANDO PROTÓTIPO COMPLETO: ML + OTIMIZAÇÃO NSGA-II 🚀")
log_geral = ["PROTÓTIPO DE OTIMIZAÇÃO DE LEITOS DE UTI - LOG DE EXECUÇÃO\n" + "="*60 + "\n"]

# ==============================================================================
# FASE 1: TREINAMENTO DO MODELO DE MACHINE LEARNING
# ==============================================================================
print("\n--- FASE 1: TREINAMENTO DO MODELO PREDITIVO (XGBOOST) ---")
log_geral.append("--- FASE 1: TREINAMENTO DO MODELO PREDITIVO (XGBOOST) ---\n")

try:
    df_ml = pd.read_csv('dataset_para_ml.csv')
    print(f"✅ Dataset 'dataset_para_ml.csv' carregado com {len(df_ml)} registros.")
    log_geral.append(f"Dataset 'dataset_para_ml.csv' carregado com {len(df_ml)} registros.\n")
except FileNotFoundError:
    print("❌ ERRO: 'dataset_para_ml.csv' não encontrado. Abortando.")
    exit()

target = 'LOS_DIAS'
features = ['IDADE', 'SEXO', 'DIAG_PRINC', 'CAR_INT', 'PROC_REA', 'MORTE']
categorical_features = ['SEXO', 'DIAG_PRINC', 'CAR_INT', 'PROC_REA', 'MORTE']

df_ml = df_ml.dropna(subset=features + [target]).copy()
for col in categorical_features:
    df_ml[col] = df_ml[col].astype(str).fillna('missing')

X = df_ml[features]
y = df_ml[target]
print(f"📊 Dados prontos para modelagem: {len(X)} registros limpos.")

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
    remainder='passthrough'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Treinando o modelo XGBoost...")
model_pipeline.fit(X_train, y_train)
print("✅ Modelo treinado!")
log_geral.append("Modelo XGBoost treinado com sucesso.\n")

y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📈 Performance do Modelo: MAE = {mae:.2f} dias | R² = {r2:.2f}")
log_geral.append(f"Performance do Modelo:\n  - MAE: {mae:.2f} dias\n  - R²: {r2:.2f}\n")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor=None)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2, label="Previsão Perfeita")
plt.title('Performance do Modelo Preditivo (LOS Real vs. Previsto)', fontsize=16)
plt.xlabel('Tempo de Permanência Real (dias)', fontsize=12)
plt.ylabel('Tempo de Permanência Previsto (dias)', fontsize=12)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'comparativo_los_real_vs_previsto.png'))
plt.close()
print(f"✅ Gráfico de performance salvo.")

joblib.dump(model_pipeline, 'modelo_los.pkl')
print("💾 Modelo de ML salvo como 'modelo_los.pkl'.\n")

# ==============================================================================
# FASE 2: CONFIGURAÇÃO DA SIMULAÇÃO E OTIMIZAÇÃO
# ==============================================================================
print("\n--- FASE 2: CONFIGURAÇÃO DA SIMULAÇÃO E OTIMIZAÇÃO ---")

B = 10
HORIZONTE_SIMULACAO_HORAS = 168
PESOS_MCDM = {'paciente': 0.6, 'eficiencia': 0.2, 'estabilidade': 0.2}

try:
    df_otimizacao = pd.read_csv('dataset_para_otimizacao.csv')
    # *** CORREÇÃO CRÍTICA: UNIR OS DATAFRAMES ***
    # Adicionar um ID único se não existir
    if 'paciente_id' not in df_ml:
      df_ml = df_ml.reset_index().rename(columns={'index': 'paciente_id'})
    if 'paciente_id' not in df_otimizacao:
      df_otimizacao = df_otimizacao.reset_index().rename(columns={'index': 'paciente_id'})

    # Unir os dois dataframes para ter todas as informações em um só lugar
    df_completo = pd.merge(df_otimizacao, df_ml.drop(columns=['LOS_DIAS']), on='paciente_id', how='inner')
    
    df_pacientes = df_completo.sample(n=40, random_state=42).copy()
    df_pacientes['tempo_chegada_ci'] = np.random.randint(0, HORIZONTE_SIMULACAO_HORAS - 48, size=len(df_pacientes))
    
    print(f"✅ Datasets unificados. Usando {len(df_pacientes)} pacientes para simulação.")
    log_geral.append(f"Datasets unificados, usando {len(df_pacientes)} pacientes.\n")
except FileNotFoundError:
    print("❌ ERRO: 'dataset_para_otimizacao.csv' não encontrado. Abortando.")
    exit()

# O restante do código permanece o mesmo, mas a lógica de previsão usará o df_completo
# ... (código da classe UTIAllocationProblem e funções auxiliares)
class UTIAllocationProblem(Problem):
    def __init__(self, pacientes_na_fila, leitos_estado, plano_anterior, hora_inicio_otimizacao):
        n_var = len(pacientes_na_fila)
        super().__init__(n_var=n_var, n_obj=3, n_constr=0, 
                         xl=np.array([p['tempo_chegada_ci'] for p in pacientes_na_fila]), 
                         xu=np.array([HORIZONTE_SIMULACAO_HORAS] * n_var), 
                         vtype=int)
        self.pacientes_na_fila = pacientes_na_fila
        self.leitos_estado_inicial = leitos_estado
        self.plano_anterior = plano_anterior
        self.hora_inicio_otimizacao = hora_inicio_otimizacao

    def _evaluate(self, x, out, *args, **kwargs):
        all_objectives = [self.calculate_objectives(sol) for sol in x]
        out["F"] = np.array(all_objectives)

    def calculate_objectives(self, solucao_tempos):
        leitos_sim = list(self.leitos_estado_inicial)
        risco_total, horas_ocupadas_futuras, mudancas = 0, 0, 0
        cronograma_atual = {}

        pacientes_ordenados = sorted(zip(solucao_tempos, self.pacientes_na_fila), key=lambda item: item[0])
        
        for tempo_admissao, paciente in pacientes_ordenados:
            leito_idx = np.argmin(leitos_sim)
            inicio_real = max(tempo_admissao, leitos_sim[leito_idx])
            duracao_horas = paciente['duracao_di_predita']
            leitos_sim[leito_idx] = inicio_real + duracao_horas
            
            cronograma_atual[paciente['paciente_id']] = {'inicio': inicio_real}
            espera = inicio_real - paciente['tempo_chegada_ci']
            risco_total += espera * paciente['gravidade_gi']
            horas_ocupadas_futuras += duracao_horas

        horizonte_restante = max(1, HORIZONTE_SIMULACAO_HORAS - self.hora_inicio_otimizacao)
        horas_ja_comprometidas = sum(max(0, leito - self.hora_inicio_otimizacao) for leito in self.leitos_estado_inicial)
        taxa_utilizacao = ((horas_ja_comprometidas + horas_ocupadas_futuras) / (B * horizonte_restante)) * 100
        obj_eficiencia = max(0, 100 - taxa_utilizacao)

        if self.plano_anterior:
            mudancas = sum(1 for pid, info in cronograma_atual.items() if pid in self.plano_anterior and self.plano_anterior.get(pid, {}).get('inicio') != info['inicio'])

        return [risco_total, obj_eficiencia, mudancas]

def run_nsga2_optimization(pacientes_na_fila, leitos_estado, plano_anterior, hora_atual):
    problem = UTIAllocationProblem(pacientes_na_fila, leitos_estado, plano_anterior, hora_atual)
    algorithm = NSGA2(pop_size=50, eliminate_duplicates=True)
    termination = get_termination("n_gen", 50)
    
    res = minimize(problem, algorithm, termination, seed=1, verbose=False)
    
    fronteira_pareto = []
    if res.X is not None:
        for i, (sol, obj) in enumerate(zip(res.X, res.F)):
            cronograma = {p['paciente_id']: {'inicio': sol[j]} for j, p in enumerate(pacientes_na_fila)}
            fronteira_pareto.append({'solucao_id': i + 1, 'cronograma': cronograma, 'objetivos': {'paciente': obj[0], 'eficiencia': obj[1], 'estabilidade': obj[2]}})
            
    return fronteira_pareto, res.F

def apply_mcdm(fronteira, pesos):
    if not fronteira: return None
    objetivos = np.array([list(sol['objetivos'].values()) for sol in fronteira])
    max_vals = np.max(objetivos, axis=0)
    max_vals[max_vals == 0] = 1
    
    norm_objetivos = objetivos / max_vals
    pesos_array = np.array(list(pesos.values()))
    
    scores = np.sum(norm_objetivos * pesos_array, axis=1)
    best_idx = np.argmin(scores)
    return fronteira[best_idx]


# ==============================================================================
# FASE 3: EXECUÇÃO DA SIMULAÇÃO DINÂMICA
# ==============================================================================
print("\n--- FASE 3: EXECUÇÃO DA SIMULAÇÃO DINÂMICA ---")

leitos_estado = [0] * B
fila_de_espera = []
plano_atual = {}
log_simulacao = []
cronograma_final_real = []

for hora_atual in range(HORIZONTE_SIMULACAO_HORAS):
    novas_chegadas = df_pacientes[df_pacientes['tempo_chegada_ci'] == hora_atual]
    
    if not novas_chegadas.empty:
        for _, paciente in novas_chegadas.iterrows():
            fila_de_espera.append(paciente.to_dict())
            log_simulacao.append({'hora': hora_atual, 'evento': 'CHEGADA', 'paciente_id': paciente['paciente_id'], 'detalhes': f"Gravidade: {paciente['gravidade_gi']}"})

    if (not novas_chegadas.empty and fila_de_espera) or (hora_atual > 0 and hora_atual % 12 == 0 and fila_de_espera):
        log_simulacao.append({'hora': hora_atual, 'evento': 'OTIMIZACAO', 'detalhes': f"Iniciando para {len(fila_de_espera)} pacientes."})
        
        for p in fila_de_espera:
            if 'duracao_di_predita' not in p:
                # *** CORREÇÃO CRÍTICA FINAL ***
                paciente_features_df = pd.DataFrame([p])[features]
                
                # Garante que as colunas e tipos são idênticos ao do treino
                for col in categorical_features:
                    paciente_features_df[col] = paciente_features_df[col].astype(str)
                
                p['duracao_di_predita'] = max(24, int(model_pipeline.predict(paciente_features_df)[0] * 24))

        fronteira, resultados_F = run_nsga2_optimization(fila_de_espera, leitos_estado, plano_atual, hora_atual)
        solucao_recomendada = apply_mcdm(fronteira, PESOS_MCDM)
        
        if solucao_recomendada:
            plano_atual = solucao_recomendada['cronograma']
            obj = solucao_recomendada['objetivos']
            log_simulacao.append({'hora': hora_atual, 'evento': 'DECISAO', 'detalhes': f"Plano #{solucao_recomendada['solucao_id']} escolhido. Risco={obj['paciente']:.0f}, Ociosidade={obj['eficiencia']:.1f}%, Mudanças={obj['estabilidade']:.0f}"})

    if plano_atual:
        pacientes_alocados_nesta_hora = []
        for paciente_id, info in plano_atual.items():
            paciente_info = next((p for p in fila_de_espera if p['paciente_id'] == paciente_id), None)
            if paciente_info and info.get('inicio') == hora_atual:
                leito_idx = np.argmin(leitos_estado)
                if leitos_estado[leito_idx] <= hora_atual:
                    duracao = paciente_info['duracao_di_predita']
                    leitos_estado[leito_idx] = hora_atual + duracao
                    log_simulacao.append({'hora': hora_atual, 'evento': 'ALOCACAO', 'paciente_id': paciente_id, 'detalhes': f"Alocado no Leito {leito_idx+1} por {duracao}h."})
                    cronograma_final_real.append({'pacienteId': paciente_id, 'leito': leito_idx, 'inicio': hora_atual, 'fim': hora_atual + duracao, 'duracaoDias': duracao/24})
                    pacientes_alocados_nesta_hora.append(paciente_id)
        
        fila_de_espera = [p for p in fila_de_espera if p['paciente_id'] not in pacientes_alocados_nesta_hora]

df_log = pd.DataFrame(log_simulacao)
df_log.to_csv(os.path.join(RESULTS_DIR, 'log_simulacao.csv'), index=False)
print("✅ Log completo da simulação salvo.")
print("🎉 SIMULAÇÃO CONCLUÍDA! 🎉\n")

# ... (O restante do código para FASE 4 permanece o mesmo)
# ==============================================================================
# FASE 4: GERAÇÃO DE RESULTADOS FINAIS
# ==============================================================================
print("\n--- FASE 4: GERAÇÃO DE RELATÓRIOS E GRÁFICOS FINAIS ---")

with open(os.path.join(RESULTS_DIR, 'relatorio_final.txt'), 'w') as f:
    f.write("RELATÓRIO FINAL DA SIMULAÇÃO\n" + "="*30 + "\n\n")
    f.write("1. PERFORMANCE DO MODELO DE MACHINE LEARNING\n")
    f.write(f"  - Erro Médio Absoluto (MAE): {mae:.2f} dias\n")
    f.write(f"  - Coeficiente de Determinação (R²): {r2:.2f}\n\n")
    
    if 'fronteira' in locals() and fronteira:
        f.write("2. ÚLTIMA FRONTEIRA DE PARETO GERADA\n")
        f.write("ID | Risco (Paciente) | Ociosidade (%) | Mudanças\n")
        for sol in fronteira:
            obj = sol['objetivos']
            f.write(f"#{sol['solucao_id']:<2} | {obj['paciente']:<16.0f} | {obj['eficiencia']:<14.1f} | {obj['estabilidade']:<9.0f}\n")
    f.write("\n")

    f.write("3. MÉTRICAS GLOBAIS DA SIMULAÇÃO\n")
    if cronograma_final_real:
        total_espera = sum(c['inicio'] - df_pacientes[df_pacientes['paciente_id'] == c['pacienteId']].iloc[0]['tempo_chegada_ci'] for c in cronograma_final_real)
        f.write(f"  - Tempo de Espera Médio: {total_espera / len(cronograma_final_real):.2f} horas\n")
        horas_ocupadas_total = sum(c['fim'] - c['inicio'] for c in cronograma_final_real)
        taxa_ocupacao_final = (horas_ocupadas_total / (B * HORIZONTE_SIMULACAO_HORAS)) * 100
        f.write(f"  - Taxa de Ocupação Final: {taxa_ocupacao_final:.2f}%\n")
    f.write(f"  - Pacientes Alocados: {len(cronograma_final_real)} de {len(df_pacientes[df_pacientes['tempo_chegada_ci'] < HORIZONTE_SIMULACAO_HORAS])}\n")

print("✅ Relatório final salvo.")

if 'resultados_F' in locals() and resultados_F is not None and len(resultados_F) > 0:
    plot = Scatter(title="Fronteira de Pareto (Risco vs. Ociosidade)", 
                   labels=["Risco Clínico (Paciente)", "Ociosidade (Eficiência)"])
    plot.add(resultados_F[:, [0,1]], s=80, facecolors='#1f77b4', edgecolors='#1f77b4', alpha=0.7)
    plot.save(os.path.join(RESULTS_DIR, "fronteira_pareto.png"))
    plt.close()
    print("✅ Gráfico 'fronteira_pareto.png' salvo.")

if cronograma_final_real:
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_pacientes)))
    
    for i in range(B): ax.axhline(i, color='gray', linestyle=':', alpha=0.5)

    for agendamento in cronograma_final_real:
        pid = agendamento['pacienteId']
        leito = agendamento['leito']
        inicio = agendamento['inicio']
        duracao = agendamento['fim'] - agendamento['inicio']
        ax.barh(y=leito, width=duracao, left=inicio, height=0.6, 
                color=colors[pid % len(colors)], edgecolor='black',
                label=f"P{pid}")
        ax.text(inicio + duracao/2, leito, f"P{pid}", ha='center', va='center', color='white', weight='bold', fontsize=8)

    ax.set_yticks(range(B))
    ax.set_yticklabels([f"Leito {i+1}" for i in range(B)])
    ax.set_xlabel("Tempo (horas)")
    ax.set_title("Cronograma de Alocação de Leitos de UTI", fontsize=16)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    max_end = max((c['fim'] for c in cronograma_final_real), default=HORIZONTE_SIMULACAO_HORAS)
    ax.set_xticks(np.arange(0, max_end + 24, 24))
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cronograma_final.png"))
    plt.close()
    print("✅ Gráfico 'cronograma_final.png' salvo.")

print("\n🎉 PROCESSO COMPLETO FINALIZADO! VERIFIQUE A PASTA 'resultados_simulacao'. 🎉")