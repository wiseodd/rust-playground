use anyhow::{Ok, Result};
use clap::Parser;

mod tch_rs {
    pub mod beam_search;
    pub mod mnist;
}

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    problem: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.problem.as_str() {
        "tch-mnist" => tch_rs::mnist::train()?,
        "tch-beamsearch" => tch_rs::beam_search::run()?,
        _ => println!("Provide an arg! Possible choices: [tch-mnist, tch-beamsearch]!"),
    }

    Ok(())
}
