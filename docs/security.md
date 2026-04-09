# Segurança — CancerGuard API

← [Voltar ao índice](./README.md)

---

## Controles de Segurança Implementados

### OWASP A03 — Injection: Validação de Input

Toda entrada da API passa pela validação Pydantic **antes** de qualquer processamento de negócio. O modelo nunca recebe dados brutos não validados.

```python
# schemas.py — exemplos de constraints
class PredictionInput(BaseModel):
    mean_radius: float = Field(..., gt=0)
    mean_smoothness: float = Field(..., gt=0, lt=1)
    # ...
```

- Tipos são forçados (`float` — strings e objetos são rejeitados)
- Ranges são verificados (`gt=0`, `lt=1` onde aplicável)
- Campos faltando retornam HTTP 422 com descrição do erro
- O modelo nunca é atingido por dados inválidos

---

### OWASP A02 — Cryptographic Failures: Gerenciamento de Segredos

| Prática | Implementação |
|---|---|
| Nenhum segredo hardcoded | Variáveis de ambiente via `.env` |
| `.env` não commitado | Listado no `.gitignore` |
| Template documentado | `.env.example` sem valores reais |
| Secrets em produção | GitHub Actions Secrets + painel do Render |

**`.gitignore` deve incluir:**
```
.env
predictions.db
mlruns/
*.joblib
```

---

### OWASP A06 — Vulnerable and Outdated Components

- `requirements.txt` com versões fixas (ex: `fastapi==0.110.0`) para builds reproduzíveis e auditáveis
- Atualizações de segurança gerenciadas via **GitHub Dependabot**:

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
```

---

### OWASP A09 — Security Logging and Monitoring Failures

- Cada predição é logada com timestamp, input e output em SQLite
- Logs de aplicação (startup, erros) vão para stdout via `logging` do Python
- O log de predições não é exposto via API — acesso restrito ao container

---

## Exposição de Dados

| Dado | Exposto via API? | Onde fica |
|---|---|---|
| Features de entrada | Não (apenas resultado) | Log interno (SQLite) |
| Probabilidade | Sim (parte do response) | Response JSON |
| Logs de predição | Não | `predictions.db` (container) |
| Credenciais | Nunca | Secrets do GitHub / Render |

---

## Limitações de Escopo

Este projeto é um portfólio de MLOps. Para uso em produção médica real, seriam necessários controles adicionais:

| Controle | Relevância |
|---|---|
| Autenticação (OAuth2 / API Key) | Necessário em produção real |
| Rate limiting | Necessário para prevenir abuso |
| HTTPS obrigatório | Render fornece TLS automaticamente |
| Criptografia do `predictions.db` | Necessário para dados sensíveis de saúde (LGPD) |
| Auditoria de acesso ao Model Registry | Necessário em ambientes regulados |
