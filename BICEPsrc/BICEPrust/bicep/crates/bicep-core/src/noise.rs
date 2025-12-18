use crate::State;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

pub struct NoiseGenerator {
    rng: ChaCha20Rng,
}

impl NoiseGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }
    
    pub fn from_path_id(global_seed: u64, path_id: u64) -> Self {
        // Combine seeds deterministically
        let seed = global_seed.wrapping_add(path_id.wrapping_mul(0x9e3779b97f4a7c15));
        Self::new(seed)
    }
    
    pub fn generate_dw(&mut self, n: usize, sqrt_dt: f64) -> State {
        let values: Vec<f64> = (0..n)
            .map(|_| {
                let sample: f64 = StandardNormal.sample(&mut self.rng);
                sample * sqrt_dt
            })
            .collect();
        State::new(values)
    }
    
    pub fn generate_antithetic_pair(&mut self, n: usize, sqrt_dt: f64) -> (State, State) {
        let dw1 = self.generate_dw(n, sqrt_dt);
        let dw2 = State(dw1.0.map(|x| -x));
        (dw1, dw2)
    }
}