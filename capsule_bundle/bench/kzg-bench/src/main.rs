use ark_bls12_381::Bls12_381;
use ark_ff::UniformRand;
use ark_poly::univariate::DensePolynomial;
use ark_poly::DenseUVPolynomial;
use ark_poly_commit::kzg10::{Commitment, KZG10, Powers, UniversalParams};
use std::borrow::Cow;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;

const NS: [usize; 3] = [1 << 20, 1 << 24, 1 << 26];
const REPEATS: usize = 3;
const FIELD_BYTES: f64 = 32.0;

fn sample_poly(n: usize, seed: u64) -> DensePolynomial<ark_bls12_381::Fr> {
    let mut rng = StdRng::seed_from_u64(seed);
    let coeffs: Vec<_> = (0..n)
        .map(|_| ark_bls12_381::Fr::rand(&mut rng))
        .collect();
    DensePolynomial::from_coefficients_vec(coeffs)
}

fn bench_commit(
    poly: &DensePolynomial<ark_bls12_381::Fr>,
    powers: &Powers<Bls12_381>,
) -> (f64, Commitment<Bls12_381>) {
    let start = Instant::now();
    let (commitment, _rand) = KZG10::commit(powers, poly, None, None).expect("commit");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    (elapsed_ms, commitment)
}

fn main() {
    println!("N,commit_ms,throughput_mb_s");
    for &n in &NS {
        let mut rng = StdRng::seed_from_u64(42);
        let degree = n.next_power_of_two();
        let params: UniversalParams<Bls12_381> =
            KZG10::<Bls12_381, DensePolynomial<_>>::setup(degree, false, &mut rng)
                .expect("setup");
        let powers_of_g = params.powers_of_g[..=degree].to_vec();
        let powers_of_gamma_g: Vec<_> = (0..=degree)
            .map(|i| params.powers_of_gamma_g[&i])
            .collect();
        let powers = Powers {
            powers_of_g: Cow::Owned(powers_of_g),
            powers_of_gamma_g: Cow::Owned(powers_of_gamma_g),
        };
        let poly = sample_poly(n, 99);

        let mut best_ms = f64::INFINITY;
        for _ in 0..REPEATS {
            let (elapsed_ms, _commit) = bench_commit(&poly, &powers);
            if elapsed_ms < best_ms {
                best_ms = elapsed_ms;
            }
        }
        let bytes = (n as f64) * FIELD_BYTES;
        let throughput_mb_s = bytes / (best_ms / 1_000.0) / (1024.0 * 1024.0);
        println!("{}, {:.3}, {:.3}", n, best_ms, throughput_mb_s);
    }
}
