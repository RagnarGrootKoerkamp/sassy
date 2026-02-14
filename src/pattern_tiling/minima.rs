#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracePostProcess {
    All,
    LocalMinima,
}

/// Find indices of local minima in a list of (position, cost) pairs
#[inline(always)]
pub fn local_minima_indices(positions_and_costs: &[(isize, isize)], out_indices: &mut Vec<usize>) {
    out_indices.clear();
    if positions_and_costs.is_empty() {
        return;
    }
    let mut prev_pos = positions_and_costs[0].0;
    let mut prev_cost = positions_and_costs[0].1;
    let mut prev_idx = 0usize;
    let mut last_trend: i8 = 2; // 2 = none, -1 = down/flat, 0 = flat, 1 = up

    for (idx, &(pos, cost)) in positions_and_costs.iter().enumerate().skip(1) {
        if pos - prev_pos > 1 {
            if last_trend != 1 {
                out_indices.push(prev_idx);
            }
            last_trend = 2;
            prev_cost = cost;
            prev_idx = idx;
            prev_pos = pos;
            continue;
        }

        let increasing = cost > prev_cost;
        let decreasing = cost < prev_cost;
        let equal = cost == prev_cost;

        if increasing && last_trend != 1 {
            out_indices.push(prev_idx);
            last_trend = 1;
        } else if decreasing {
            last_trend = -1;
        } else if equal && last_trend == 2 {
            last_trend = 0;
        }

        prev_cost = cost;
        prev_idx = idx;
        prev_pos = pos;
    }

    if last_trend != 1 {
        out_indices.push(prev_idx);
    }
}
