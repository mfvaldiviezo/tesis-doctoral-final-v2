import logging
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

logger = logging.getLogger("tsc.h_sarg")

class HSARGExtractor(BaseFeaturesExtractor):
    """
    H-SARG (Hybrid Self-Attention Gated Risk) Features Extractor
    ============================================================
    Tesis Doctoral: Control Semafórico Inteligente con RL Sensible al Riesgo
    Capítulo 4.4 — Arquitectura de Explicabilidad y Toma de Decisiones

    Esta clase implementa una red neuronal profunda personalizada para PPO en PyTorch.
    En lugar de un perceptrón multicapa ordinario (caja negra), H-SARG descompone el
    vector de estado s_t ∈ ℝ^34 en dos ramas y aplica auto-atención multi-cabezal (MHSA)
    e interpolación sigmoide para aislar y jerarquizar de forma transparente las variables de riesgo.

    Descomposición de s_t ∈ ℝ^34:
      - Rama de Riesgo (Indices 0 - 23):
        - Colas vehiculares (q_t ∈ ℝ^12)
        - Tiempos de espera (w_t ∈ ℝ^12)
      - Rama Nominal (Indices 24 - 33):
        - Presión espacial proyectada (p_t ∈ ℝ^4)
        - Codificación One-Hot de Fase activa (phi_t ∈ ℝ^4)
        - Edad y edad normalizada de fase (tau_t ∈ ℝ^2)
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # Validar dimensión del espacio de observaciones
        obs_dim = observation_space.shape[0]
        assert obs_dim == 34, f"H-SARG requiere un vector de estado de exactamente 34-D. Recibido: {obs_dim}"

        # ── 1. DIMENSIONAMIENTO DE RAMAS ──────────────────────────────────────
        self.risk_in_dim = 24  # 12 queues + 12 waits
        self.nominal_in_dim = 10  # 4 pressures + 4 phases + 2 ages
        
        # ── 2. RAMA NOMINAL (Procesamiento Lineal Simple) ─────────────────────
        self.nominal_net = nn.Sequential(
            nn.Linear(self.nominal_in_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        # ── 3. RAMA DE RIESGO CON AUTO-ATENCIÓN MULTI-CABEZAL (MHSA) ──────────
        # Mapeamos los 12 carriles a una representación latente de dimensión d_model=16
        # Cada carril i tiene 2 características: cola[i] y espera[i]
        self.carril_embed = nn.Linear(2, 16)
        
        # Capa de Auto-Atención Multi-Cabezal (2 cabezales para relaciones espaciales)
        self.self_attention = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
        
        # ── 4. COMPUERTA SIGMOIDE ABSOLUTA (Sigmoid Gating) ───────────────────
        # Mapea las características atendidas a un factor de riesgo elástico [0, 1]
        self.risk_gating = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Reducción dimensional del riesgo atendido y filtrado
        self.risk_reducer = nn.Sequential(
            nn.Linear(12 * 16, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # ── 5. CABEZAL DE FUSIÓN DE CARACTERÍSTICAS (Nominal + Riesgo Gated) ──
        self.fusion_net = nn.Sequential(
            nn.Linear(64 + 32, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
        
        # Buffer en CPU para extraer los pesos de atención en tiempo real (XAI)
        self.last_attention_weights = None

        logger.info("🧠 Arquitectura H-SARG inicializada con éxito en PyTorch.")
        logger.info("   Rama Nominal: 10-D -> LayerNorm(32)")
        logger.info("   Rama Riesgo : 24-D -> Embed(16) -> MHSA(2 Heads) -> Sigmoid Gating")
        logger.info("   Dimensión Final Fusionada: %d", features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extraer dimensiones de lote
        batch_size = observations.shape[0]

        # ── A. Segmentación del Vector de Estados s_t ──────────────────────────
        # Rama de Riesgo: Colas (0-11) y Esperas (12-23)
        queues = observations[:, 0:12]  # (Batch, 12)
        waits = observations[:, 12:24]  # (Batch, 12)
        
        # Rama Nominal: Presión (24-27), Fase (28-31), Edad (32-33)
        nominal_features = observations[:, 24:34]  # (Batch, 10)

        # ── B. Procesar Rama Nominal ──────────────────────────────────────────
        nominal_latent = self.nominal_net(nominal_features)  # (Batch, 32)

        # ── C. Procesar Rama de Riesgo con MHSA ────────────────────────────────
        # Dar forma a la entrada para la atención: 12 carriles, 2 features cada uno
        # shape: (Batch, 12, 2)
        risk_grouped = torch.stack([queues, waits], dim=-1)
        
        # Proyectar a dimensión d_model=16: (Batch, 12, 16)
        risk_embed = self.carril_embed(risk_grouped)
        
        # Aplicar Auto-Atención Multi-Cabezal (MHSA)
        # attn_output: representación relacional de cada carril
        # attn_weights: matriz de atención 12x12 de explicabilidad
        attn_output, attn_weights = self.self_attention(risk_embed, risk_embed, risk_embed)
        
        # Persistir pesos de atención para visualización XAI (se extrae el promedio del lote)
        self.last_attention_weights = attn_weights.detach().cpu().numpy()

        # ── D. Aplicar Compuerta de Riesgo (Sigmoid Gating) ───────────────────
        # Calcula una puntuación de riesgo elástica independiente por cada carril
        gates = self.risk_gating(attn_output)  # (Batch, 12, 1)
        
        # Multiplicación elemento a elemento (Gating): filtra las características atendidas
        gated_risk = attn_output * gates  # (Batch, 12, 16)

        # Aplanar y reducir la rama de riesgo
        gated_risk_flat = gated_risk.reshape(batch_size, -1)  # (Batch, 12 * 16)

        risk_latent = self.risk_reducer(gated_risk_flat)   # (Batch, 64)

        # ── E. Fusión y Proyección Final ──────────────────────────────────────
        combined = torch.cat([nominal_latent, risk_latent], dim=-1)  # (Batch, 32 + 64 = 96)
        features = self.fusion_net(combined)  # (Batch, features_dim)

        return features

    def get_explainability_weights(self) -> torch.Tensor:
        """Retorna la última matriz de auto-atención calculada para auditoría de tráfico."""
        return self.last_attention_weights


class CoopSARGExtractor(BaseFeaturesExtractor):
    """
    Coop-SARG (Cooperative Self-Attention Gated Risk) Features Extractor
    ===================================================================
    Tesis Doctoral: Control Semafórico Inteligente con RL Sensible al Riesgo
    Propuesta de Extensión Novedosa: Mitigación del Efecto Braess bajo Caos Severo

    Esta arquitectura se presenta como una extensión multiobjetivo de H-SARG orientada a integrar:
      1. PILAR I (EQUIDAD DISTRIBUTIVA): Mecanismo de modulación adaptativa basado en la desigualdad
         de espera local (desviación absoluta frente a la media) para reducir el riesgo de inanición.
      2. PILAR II (EXPLICABILIDAD DUAL): Separación explícita mediante auditoría dual de la auto-atención
         espacial relacional de carriles (12x12) y las compuertas cooperativas de supresión downstream (12x1).
      3. PILAR III (RESILIENCIA ADAPTATIVA): Escalamiento dinámico de sensibilidad exponencial ante
         las colas críticas locales como condiciones precursoras de gridlock vehicular.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        assert obs_dim == 34, f"Coop-SARG requiere un vector de estado de exactamente 34-D. Recibido: {obs_dim}"

        self.risk_in_dim = 24  # 12 queues + 12 waits
        self.nominal_in_dim = 6  # 4 phases + 2 ages (pressures are moved to risk branch)
        
        # Rama Nominal Reducida (phases y ages)
        self.nominal_net = nn.Sequential(
            nn.Linear(self.nominal_in_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )

        # Rama de Riesgo Cooperativo: [cola, espera, presión_aguas_abajo]
        self.carril_embed = nn.Linear(3, 16)
        
        # Auto-Atención Multi-Cabezal Cooperativa (2 Heads)
        self.self_attention = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
        
        # Compuerta Sigmoide Cooperativa (Downstream Gating)
        self.risk_gating = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Reductor de la Rama de Riesgo Gated
        self.risk_reducer = nn.Sequential(
            nn.Linear(12 * 16, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # Cabezal de Fusión
        self.fusion_net = nn.Sequential(
            nn.Linear(64 + 16, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
        
        # Buffers de explicabilidad dual para auditorías en tiempo real (XAI)
        self.last_attention_weights = None
        self.last_cooperative_gates = None

        logger.info("✨ Nueva arquitectura Coop-SARG con Garantía de Equidad, Explicabilidad y Resiliencia inicializada.")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # ── A. Segmentación del Vector de Estados ──────────────────────────────
        queues = observations[:, 0:12]      # (Batch, 12)
        waits = observations[:, 12:24]      # (Batch, 12)
        pressures = observations[:, 24:28]  # (Batch, 4) — Presión por acceso (N, S, E, O)
        nominal_features = observations[:, 28:34]  # (Batch, 6) — 4 phases + 2 ages

        # ── B. Procesar Rama Nominal Reducida ──────────────────────────────────
        nominal_latent = self.nominal_net(nominal_features)  # (Batch, 16)

        # ── C. Resiliencia: Amplificación Exponencial de Presión Cooperativa ────
        # Si la cola máxima local se acerca al gridlock, amplificamos la sensibilidad a la
        # presión de salida para forzar la dosificación perimetral defensiva
        max_q, _ = torch.max(queues, dim=-1, keepdim=True)  # (Batch, 1)
        resilience_scale = torch.exp(max_q / 10.0)  # Amplificación suave basada en la saturación
        pressures_resilient = pressures * resilience_scale

        # Mapeamos las 4 presiones viales a los 12 carriles (repitiendo cada presión 3 veces)
        lane_pressure = torch.repeat_interleave(pressures_resilient, repeats=3, dim=-1)  # (Batch, 12)

        # ── D. Rama de Riesgo Cooperativo con MHSA ─────────────────────────────
        # Stacking: [Cola local, Espera local, Presión de salida resiliente]
        risk_grouped = torch.stack([queues, waits, lane_pressure], dim=-1)  # (Batch, 12, 3)
        
        risk_embed = self.carril_embed(risk_grouped)  # (Batch, 12, 16)
        
        # MHSA Cooperativo
        attn_output, attn_weights = self.self_attention(risk_embed, risk_embed, risk_embed)
        self.last_attention_weights = attn_weights.detach().cpu().numpy()

        # ── E. Compuerta Sigmoide Cooperativa (Downstream Gating) ──────────────
        gates = self.risk_gating(attn_output)  # (Batch, 12, 1)
        self.last_cooperative_gates = gates.detach().cpu().numpy()

        # ── F. Garantía de Equidad (Gini-equity Modulation) ────────────────────
        # Calculamos la desviación absoluta de los tiempos de espera respecto a la media
        mean_waits = waits.mean(dim=-1, keepdim=True)  # (Batch, 1)
        wait_deviation = waits - mean_waits  # Desviación local (Batch, 12)
        gini_approx = torch.abs(wait_deviation).mean(dim=-1, keepdim=True)  # Desviación absoluta media (Batch, 1)
        
        # Boost de equidad: si una cola ha esperado significativamente más que la media,
        # e injusticia general (gini_approx) es alta, aumentamos su gate para evitar inanición
        equity_boost = torch.sigmoid(wait_deviation * gini_approx)  # (Batch, 12)
        
        # Modulación conjunta de compuertas
        gated_risk = attn_output * (gates * equity_boost.unsqueeze(-1))  # (Batch, 12, 16)

        # Reducir la rama de riesgo
        gated_risk_flat = gated_risk.reshape(batch_size, -1)  # (Batch, 12 * 16)
        risk_latent = self.risk_reducer(gated_risk_flat)   # (Batch, 64)

        # ── G. Fusión Cooperativa Final ────────────────────────────────────────
        combined = torch.cat([nominal_latent, risk_latent], dim=-1)  # (Batch, 16 + 64 = 80)
        features = self.fusion_net(combined)  # (Batch, features_dim)

        return features

    def get_explainability_weights(self) -> dict:
        """Retorna el sistema de auditoría dual para visualizaciones XAI completas."""
        return {
            "spatial_attention_12x12": self.last_attention_weights,
            "downstream_gating_12x1": self.last_cooperative_gates
        }
