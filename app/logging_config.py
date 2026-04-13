import sqlite3
import json
import os
from datetime import datetime


class PredictionLogger:
    """
    Logger de auditoria que persiste cada predição em um banco SQLite.

    Por que SQLite e não logs em arquivo?
    SQLite permite consultas estruturadas sobre o histórico de predições:
    filtrar por data, contar por classe, calcular médias de probabilidade.
    Isso é a base para detectar data drift no futuro (Fase de Monitoramento).

    Por que check_same_thread=False?
    O FastAPI processa requisições de forma assíncrona, o que significa que
    diferentes corrotinas podem chamar o logger de threads distintas. O SQLite,
    por padrão, impede isso. check_same_thread=False remove essa restrição —
    é seguro aqui porque o logger é um singleton e as escritas são sequenciais.

    Por que singleton no startup?
    Criar uma nova conexão SQLite a cada requisição seria ineficiente e
    arriscaria condições de concorrência. Instanciado uma vez no lifespan
    da FastAPI e reutilizado em todas as requisições.
    """

    def __init__(self, db_path: str = None):
        # Prioridade: argumento → variável de ambiente → valor padrão
        self.db_path = db_path or os.getenv("LOG_DB_PATH", "predictions.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self) -> None:
        """
        Cria a tabela se ainda não existir (IF NOT EXISTS = idempotente).
        Chamado no __init__, então a tabela está sempre pronta antes
        da primeira requisição.
        """
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                timestamp   TEXT,
                input_data  TEXT,
                prediction  TEXT,
                probability REAL
            )
        """)
        self.conn.commit()

    def log(self, input_data: dict, prediction: str, probability: float) -> None:
        """
        Persiste uma predição.

        input_data é serializado como JSON (TEXT) porque SQLite não tem
        tipo nativo para dicionários. Para consultar um campo específico
        depois: json_extract(input_data, '$.mean_radius')
        """
        self.conn.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?)",
            (
                datetime.now().isoformat(),  # ex: "2026-04-10T14:32:01.123456"
                json.dumps(input_data),
                prediction,
                probability,
            ),
        )
        self.conn.commit()
