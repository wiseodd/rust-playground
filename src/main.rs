use anyhow::{Ok, Result};
use clap::Parser;

mod mnist;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    problem: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.problem.as_str() {
        "mnist" => mnist::train()?,
        _ => println!("Provide an arg! Possible choices: [mnist]!"),
    }

    Ok(())
}
