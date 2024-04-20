use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use tch::nn::{self, ModuleT, OptimizerConfig};

fn load_net(vs: &nn::Path) -> impl nn::Module {
    nn::seq()
        .add(nn::linear(vs / "fc1", 784, 100, Default::default()))
        .add_fn(|xs| xs.relu()) // this is a lambda!
        .add(nn::linear(vs / "fc2", 100, 10, Default::default()))
}

pub fn train() -> Result<()> {
    let data = tch::vision::mnist::load_dir("./data/mnist")?;

    let vs = nn::VarStore::new(tch::Device::Mps);
    let model = load_net(&vs.root());

    let mut opt = nn::AdamW::default().build(&vs, 1e-3)?;

    const N_EPOCHS: u64 = 100;
    let pbar = ProgressBar::new(N_EPOCHS);
    pbar.set_style(ProgressStyle::with_template("[{msg}] {wide_bar}")?);

    for epoch in 1..N_EPOCHS {
        let mut avg_loss = 0f32;

        for (x, y) in data.train_iter(128).shuffle().to_device(vs.device()) {
            let out = model.forward_t(&x, true);
            let loss = out.cross_entropy_for_logits(&y);

            opt.backward_step(&loss);
            opt.zero_grad();

            avg_loss = 0.9 * avg_loss + 0.1 * f32::try_from(loss)?;
        }

        let test_acc =
            model.batch_accuracy_for_logits(&data.test_images, &data.test_labels, vs.device(), 256);
        let test_acc = test_acc * 100.;

        pbar.set_message(format!(
            "Epoch: {epoch:4}; Train loss: {avg_loss:5.3}; Test acc: {test_acc:5.2}%",
        ));
        pbar.inc(1);
    }

    Ok(())
}
