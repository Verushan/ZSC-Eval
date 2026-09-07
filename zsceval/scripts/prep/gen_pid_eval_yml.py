#!/usr/bin/env python
"""Build the evaluation yml for one partner-conditioning ladder rung.

The ladder asks whether *knowing who you are playing with* helps at all. That
question is answered against the population the rung trained on, with the true
ids, and not against held-out partners: a held-out partner has no id the agent
ever saw, so scoring a rung there measures the handicap of a useless input
rather than the value of a useful one. The held-out number is still worth
having -- it is the ZSC result, and rungs 2 and 3 are expected to do no better
there -- but it cannot establish the ceiling.

So the pool here is exactly the stage-2 training yml, with the trainable
`fcp_adaptive` entry replaced in place by the trained rung checkpoint.

*In place* is the whole trick. `policy_pool.load_population` assigns
`id = (i + 1) / num_policies` by position, so the ids a rung learned are a
property of the ordering in the training yml. Appending the rung at the end, or
dropping the placeholder, renumbers every partner and the agent is then read
its training ids shifted by one -- which would look exactly like "knowing the
partner does not help".

The rung entry points at a `*_policy_config_pid{DIM}.pkl` written by
`store_policy_config.py --pid_obs_dim DIM`, because its actor observation is
wider than the partners': +1 channel for the raw scalar, +DIM for the one-hot.
`env_policy` trims each partner back to the width it was built for.

    python prep/gen_pid_eval_yml.py random0 \
        --train_yml random0/fcp/s2/train-bench_sp.yml \
        --actor random0/fcp/s2/fcp-S2-bench_sp-rung3/1.pt \
        --pid_obs_dim 10 --out random0/pid_ladder/rung3_s1.yml
"""

import argparse
import os
import os.path as osp

import yaml
from loguru import logger

POLICY_POOL_DIR = os.getenv("POLICY_POOL")
EGO = "fcp_adaptive"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("layout")
    ap.add_argument("--train_yml", required=True, help="Pool-relative stage-2 training yml.")
    ap.add_argument("--actor", required=True, help="Pool-relative rung checkpoint.")
    ap.add_argument(
        "--pid_obs_dim",
        type=int,
        default=None,
        help="Width the rung was trained at. Omit for rung 1, whose actor never "
        "saw the id and so uses the ordinary rnn config.",
    )
    ap.add_argument("--out", required=True, help="Pool-relative output path.")
    ap.add_argument("--ego_name", default=EGO)
    args = ap.parse_args()

    assert POLICY_POOL_DIR, "POLICY_POOL is unset; source .env first"
    train_path = osp.join(POLICY_POOL_DIR, args.train_yml)
    with open(train_path) as f:
        # yaml.safe_load preserves insertion order in py3.7+, which is what the
        # id assignment depends on.
        pool = yaml.safe_load(f)

    assert args.ego_name in pool, f"{args.train_yml} has no {args.ego_name} entry"
    actor_abs = osp.join(POLICY_POOL_DIR, args.actor)
    assert osp.isfile(actor_abs), f"missing checkpoint {actor_abs}"

    suffix = "" if args.pid_obs_dim is None else f"_pid{args.pid_obs_dim}"
    config_rel = f"{args.layout}/policy_config/rnn_policy_config{suffix}.pkl"
    config_abs = osp.join(POLICY_POOL_DIR, config_rel)
    assert osp.isfile(config_abs), (
        f"missing {config_abs}. Write it with "
        f"`prep/store_policy_config.py {args.layout} --pid_obs_dim {args.pid_obs_dim}`"
    )

    position = list(pool).index(args.ego_name)
    pool[args.ego_name] = {
        "policy_config_path": config_rel,
        "featurize_type": "ppo",
        "train": False,
        "model_path": {"actor": args.actor},
    }

    out_abs = osp.join(POLICY_POOL_DIR, args.out)
    os.makedirs(osp.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w") as f:
        yaml.safe_dump(pool, f, sort_keys=False)

    partners = [k for k in pool if k != args.ego_name]
    logger.success(
        f"wrote {out_abs}: {args.ego_name} at position {position} of {len(pool)} "
        f"(id {(position + 1) / len(pool):.4f}), {len(partners)} partners"
    )
    print(out_abs)


if __name__ == "__main__":
    main()
