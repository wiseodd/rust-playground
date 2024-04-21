use anyhow::{Ok, Result};
use ndarray::Array1;

fn midpoints(n: u32, low: f32, high: f32) -> Array1<f32> {
    let step = (high - low) / (n as f32);
    let start = low + 0.5 * step;
    let end = high + 0.5 * step;
    Array1::range(start, end, step)
}

fn midpoint_rule<F>(f: F, low: f32, high: f32, n: u32) -> f32
where
    F: FnMut(f32) -> f32,
{
    let x: Array1<f32> = midpoints(n, low, high);
    let fx: Array1<f32> = x.mapv_into(f);
    ((high - low) / (n as f32) * fx).sum()
}

pub fn run() -> Result<()> {
    let low: f32 = -3.;
    let high: f32 = 3.;
    let n: u32 = 1000;

    fn f(x: f32) -> f32 {
        (-(3. * x).sin().powf(2.) - x.powf(2.)).exp()
    }

    let integral: f32 = midpoint_rule(f, low, high, n);
    println!("Integral estimate with the midpoint rule: {integral:.5}");

    Ok(())
}
