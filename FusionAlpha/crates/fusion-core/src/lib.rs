pub mod graph;
pub mod propagation;
pub mod priors;
pub mod actions;

pub use graph::{Graph, NodeFeat, Edge};
pub use propagation::{propagate_committor, PropConfig};
pub use priors::{Priors, PriorSource};
pub use actions::{ActionDecoder, pick_next_node};

/// Core types
pub type NodeId = usize;
pub type F = f32;
pub type Severity = f32; // [0,1] from ENN
pub type Committor = f32; // [0,1] value

/// Integration with BICEP/ENN
#[derive(Clone, Debug)]
pub struct FusionState {
    pub q_prior_enn: F,     // ENN's q prediction
    pub severity: Severity,  // ENN contradiction level
    pub bicep_confidence: F, // BICEP path reliability
}

impl FusionState {
    pub fn new(q_prior_enn: F, severity: Severity, bicep_confidence: F) -> Self {
        Self { q_prior_enn, severity, bicep_confidence }
    }
    
    pub fn propagation_steps(&self, t_max: usize) -> usize {
        1 + ((self.severity * t_max as F) as usize).min(t_max)
    }
}