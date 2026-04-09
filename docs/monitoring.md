# Monitoramento e Logging — CancerGuard API

← [Voltar ao índice](./README.md)

---

## 1. Estratégia de Logging

O sistema usa dois níveis complementares de logging:

| Nível | Instrumento | O que registra | Destino |
|---|---|---|---|
| Aplicação | Python `logging` | Startup, erros, warnings | stdout / stderr |
| Predições | `PredictionLogger` (SQLite) | Cada predição com input e output | `predictions.db` |

---

## 2. Logging de Aplicação (`logging_config.py`)

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("cancerguard")
```

Eventos logados:
- Startup da API e carregamento do modelo
- Erros de predição (`ERROR`)
- Cada requisição ao `/predict` (`INFO`)

---

## 3. Auditoria de Predições (`PredictionLogger`)

```python
# app/logging_config.py
import sqlite3
import json
from datetime import datetime

class PredictionLogger:
    def __init__(self, db_path: str = "predictions.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                timestamp  TEXT,
                input_data TEXT,
                prediction TEXT,
                probability REAL
            )
        """)
        self.conn.commit()

    def log(self, input_data: dict, prediction: str, probability: float):
        self.conn.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), json.dumps(input_data), prediction, probability)
        )
        self.conn.commit()
```

**Instanciação:** O `PredictionLogger` é criado como singleton no evento de startup da FastAPI para evitar múltiplas conexões SQLite concorrentes.

---

## 4. Schema da Tabela `predictions`

| Coluna | Tipo SQLite | Exemplo |
|---|---|---|
| `timestamp` | TEXT | `"2026-04-02T14:32:01.123456"` |
| `input_data` | TEXT | `"{\"mean_radius\": 14.0, ...}"` |
| `prediction` | TEXT | `"malignant"` |
| `probability` | REAL | `0.83` |

### Exemplo de registro

```json
{
  "timestamp": "2026-04-02T14:32:01.123456",
  "input_data": "{\"mean_radius\": 14.0, \"mean_texture\": 19.0, ...}",
  "prediction": "malignant",
  "probability": 0.83
}
```

---

## 5. Consultando os Logs

```bash
# Acessar o container em produção
docker exec -it cancerguard-api-container sqlite3 predictions.db

# Consultas úteis
SELECT COUNT(*) FROM predictions;
SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction;
SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10;
SELECT AVG(probability) FROM predictions WHERE prediction = 'malignant';
```

---

## 6. Roadmap de Observabilidade (Futuro)

Para escalar o monitoramento além do escopo atual de portfólio:

| Métrica | Instrumento | Valor |
|---|---|---|
| Latência de predição (p50, p95, p99) | Prometheus + Grafana | Detectar degradação |
| Taxa de requisições por minuto | Prometheus | Capacity planning |
| Distribuição de predições (% maligno) | Dashboard Grafana | Detectar data drift |
| Data drift nas features de entrada | Evidently AI | Alertar retreinamento |
| Erros HTTP (4xx, 5xx) | Render logs + alertas | SLA monitoring |

### Exemplo de integração Prometheus (futuro)

```python
# Adicionar ao main.py
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
# Expõe métricas em GET /metrics
```
