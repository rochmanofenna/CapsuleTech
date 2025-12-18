pub mod state;
pub mod drift;
pub mod diffusion;
pub mod integrators;
pub mod path;
pub mod noise;
pub mod measure;

// Core types
pub type F = f64;
pub use state::{State, Time};
pub use noise::NoiseGenerator;

// SDE traits
pub use drift::Drift;
pub use diffusion::Diffusion;

// Integrators
pub use integrators::{SdeIntegrator, Calc, EulerMaruyama, Milstein, HeunStratonovich};

// Path and ensemble types
pub use path::{Path, PathSpec, Ensemble};

// Convenience aliases for SDE-specific usage
pub type SdePath = Path<State>;
pub type SdeEnsemble = Ensemble<State>;
