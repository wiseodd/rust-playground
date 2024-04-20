use anyhow::{Ok, Result};
use tch::{kind, IndexOp, Kind, Tensor};

pub fn run() -> Result<()> {
    const SEQ_LEN: i64 = 10;
    const N_OPTS: i64 = 1_000;

    let transition_logprobs =
        Tensor::randn([SEQ_LEN, N_OPTS], kind::FLOAT_CPU).log_softmax(-1, Kind::Float);

    // Greedy
    let mut res = Vec::new();

    for i in 0..transition_logprobs.size()[0] {
        let idx_best = i64::try_from(transition_logprobs.i(i).argmax(0, false))?;
        res.push(idx_best);
    }

    println!();
    println!("Greedy: {res:?}");
    println!();

    // Beam Search
    const BEAM_WIDTH: i64 = 3;
    let mut cand_seqs: Vec<Vec<i64>> = vec![vec![], vec![], vec![]];
    let mut cand_logprobs = Tensor::from_slice(&[0., 0., 0.]);

    // Initialize sequence candidates (t = 0)
    let (topk_vals, topk_idxs) = transition_logprobs.i(0).topk(BEAM_WIDTH, 0, true, true);

    cand_logprobs += topk_vals;

    for k in 0..BEAM_WIDTH as usize {
        let k_i64 = i64::try_from(k)?;
        cand_seqs[k].push(i64::try_from(topk_idxs.i(k_i64))?);
    }

    // Rest of sequences
    for t in 1..transition_logprobs.size()[0] as i64 {
        let logprobs = transition_logprobs.i(t);
        // (beam_width, n_opts)
        let cand_logprobs_beam = cand_logprobs
            .unsqueeze(1)
            .expand([BEAM_WIDTH, N_OPTS], true);
        // (beam_width*n_opts)
        let total_logprobs = (cand_logprobs_beam + logprobs.unsqueeze(0)).flatten(0, -1);
        // Each (beam_width,)
        let (topk_vals, topk_idxs) = total_logprobs.topk(BEAM_WIDTH, 0, true, true);

        // Convert flattened indices to the real indices (beam_width,)
        let topk_seq_idxs = topk_idxs.remainder(N_OPTS);
        // Infer which text_cand each topk_token is the continuation of
        let topk_beam_idxs = topk_idxs.floor_divide_scalar(N_OPTS);

        // println!("{N_OPTS} {topk_seq_idxs:?} {topk_beam_idxs:?}");

        // Concatenate the new topk indices to the
        for k in 0..BEAM_WIDTH as usize {
            let k_i64 = i64::try_from(k)?;
            let beam_idx = i64::try_from(topk_beam_idxs.i(k_i64))?;
            let beam_idx = usize::try_from(beam_idx)?;

            let mut new_cand = cand_seqs[beam_idx].clone();
            new_cand.push(i64::try_from(topk_seq_idxs.i(k_i64))?);
            cand_seqs[k] = new_cand;
        }

        // The associated logprobs
        cand_logprobs = topk_vals;
    }

    println!("Beam Search:");

    for k in 0..BEAM_WIDTH as usize {
        let k_i64 = i64::try_from(k)?;
        let logprob = f32::try_from(cand_logprobs.i(k_i64))?;
        let cand_seq = cand_seqs[k].clone();
        println!("Candidate {} (logprob: {logprob:.5}): {cand_seq:?}", k + 1);
    }

    Ok(())
}
