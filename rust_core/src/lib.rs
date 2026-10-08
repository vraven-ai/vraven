//! Optional VRAVEN CPU acceleration primitives.
//!
//! The Python implementation remains the reference implementation. A routine
//! should move here only after profiling demonstrates a measurable bottleneck and
//! parity tests prove numerically equivalent results.

use rayon::prelude::*;

/// Compact descriptive statistics for one activation vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivationSummary {
    pub mean: f32,
    pub variance: f32,
    pub energy: f32,
    pub l1: f32,
    pub sparsity: f32,
}

/// Compute reduced activation statistics using deterministic parallel reductions.
pub fn summarise(values: &[f32], epsilon: f32) -> ActivationSummary {
    if values.is_empty() {
        return ActivationSummary {
            mean: 0.0,
            variance: 0.0,
            energy: 0.0,
            l1: 0.0,
            sparsity: 0.0,
        };
    }
    let count = values.len() as f32;
    let sum: f32 = values.par_iter().copied().sum();
    let mean = sum / count;
    let variance: f32 = values
        .par_iter()
        .map(|value| {
            let centred = *value - mean;
            centred * centred
        })
        .sum::<f32>()
        / count;
    let energy: f32 = values.par_iter().map(|value| value * value).sum();
    let l1: f32 = values.par_iter().map(|value| value.abs()).sum();
    let sparse = values
        .par_iter()
        .filter(|value| value.abs() < epsilon)
        .count() as f32;
    ActivationSummary {
        mean,
        variance,
        energy,
        l1,
        sparsity: sparse / count,
    }
}

/// Cosine similarity with explicit handling of zero vectors.
pub fn cosine_similarity(left: &[f32], right: &[f32], epsilon: f32) -> Result<f32, &'static str> {
    if left.len() != right.len() {
        return Err("vector lengths differ");
    }
    let dot: f32 = left
        .par_iter()
        .zip(right.par_iter())
        .map(|(a, b)| a * b)
        .sum();
    let left_norm: f32 = left.par_iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm: f32 = right.par_iter().map(|value| value * value).sum::<f32>().sqrt();
    let denominator = left_norm * right_norm;
    if denominator <= epsilon {
        let equal = left
            .iter()
            .zip(right.iter())
            .all(|(a, b)| (*a - *b).abs() <= epsilon);
        return Ok(if equal { 1.0 } else { 0.0 });
    }
    Ok(dot / denominator)
}

/// Deterministically resample a vector to a fixed fingerprint width.
pub fn linear_resample(values: &[f32], dimensions: usize) -> Result<Vec<f32>, &'static str> {
    if dimensions == 0 {
        return Err("dimensions must be greater than zero");
    }
    if values.is_empty() {
        return Ok(vec![0.0; dimensions]);
    }
    if values.len() == dimensions {
        return Ok(values.to_vec());
    }
    if dimensions == 1 {
        return Ok(vec![values[0]]);
    }
    let source_last = values.len() - 1;
    let target_last = dimensions - 1;
    let mut output = Vec::with_capacity(dimensions);
    for target_index in 0..dimensions {
        let position = target_index as f32 * source_last as f32 / target_last as f32;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        if lower == upper {
            output.push(values[lower]);
        } else {
            let fraction = position - lower as f32;
            output.push(values[lower] * (1.0 - fraction) + values[upper] * fraction);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_matches_known_values() {
        let result = summarise(&[1.0, -1.0, 0.0, 2.0], 1e-8);
        assert!((result.mean - 0.5).abs() < 1e-6);
        assert!((result.energy - 6.0).abs() < 1e-6);
        assert!((result.sparsity - 0.25).abs() < 1e-6);
    }

    #[test]
    fn resample_has_requested_width() {
        let output = linear_resample(&[0.0, 1.0, 2.0], 7).unwrap();
        assert_eq!(output.len(), 7);
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[6] - 2.0).abs() < 1e-6);
    }
}
