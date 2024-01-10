use std::ops::{Add, Div};

pub(crate) fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32]) {
    let ss = rmsnorm_reduce(x);
    for ((o, &w), &x) in o.iter_mut().zip(x).zip(weight) {
        *o = w * (ss * x);
    }
}

pub(crate) fn rmsnorm_inplace(x: &mut [f32], weight: &[f32]) {
    // ss = (Σx^2 / n + δ)^(-1/2)
    let ss = rmsnorm_reduce(x);
    for (x, &w) in x.iter_mut().zip(weight) {
        *x = w * (ss * *x);
    }
}

#[inline]
fn rmsnorm_reduce(x: &[f32]) -> f32 {
    // (Σx^2 / n + δ)^(-1/2)
    x.iter()
        .map(|x| x * x)
        .sum::<f32>()
        .div(x.len() as f32)
        .add(1e-5)
        .sqrt()
        .recip()
}

pub(crate) fn matmul(xout: &mut [f32], x: &[f32], w: &[f32]) {
    let n = x.len();
    xout.iter_mut().enumerate().for_each(|(i, y)| {
        *y = w[i * n..].iter().zip(x).map(|(&w, &x)| w * x).sum::<f32>();
    });
}

pub(crate) fn softmax(x: &mut [f32]) {
    let max = *x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let sum = x
        .iter_mut()
        .map(|x| {
            *x = (*x - max).exp();
            *x
        })
        .sum::<f32>();
    x.iter_mut().for_each(|x| *x /= sum);
}
