use std::path::PathBuf;

#[derive(clap::Parser)]
pub struct CrisprArgs {
    // Named args
    /// Path to file with guide sequences (including PAM)
    #[arg(long, short = 'g')]
    pub(super) guide: String,

    /// Report matches up to (and including) this distance threshold (excluding PAM).
    #[arg(short, long)]
    pub(super) k: usize,

    // optional
    /// Output file, otherwise stdout.
    #[arg(short = 'o', long)]
    pub(super) output: Option<PathBuf>,

    /// Allow at most max_n_frac of N bases in the target sequence. Values must be in the
    /// range [0. 1].  A value of 0 will allow only hits where the target sequence contains no
    /// Ns. A value of 0.1-0.2 will allow for matches that include a small number of Ns.
    #[arg(long)]
    pub(super) max_n_frac: f32,

    /// Number of threads to use. All CPUs by default.
    #[arg(short = 'j', long)]
    pub(super) threads: Option<usize>,

    /// The length of the PAM.
    #[arg(long, default_value_t = 3)]
    pub(super) pam_length: usize,

    // Flags
    /// Allow edits in PAM sequence.
    #[arg(long)]
    pub(super) allow_pam_edits: bool,

    /// Disable reverse complement search.
    #[arg(long)]
    pub(super) no_rc: bool,

    /// Use the v2 encoded search. All guides must have the same length.
    #[arg(long)]
    pub(super) v2: bool,

    /// Fasta file to search. May be gzipped.
    pub(super) path: PathBuf,
}

pub fn crispr(args: &CrisprArgs) {
    crate::grep::SearchArgs::run_crispr(args);
}

#[cfg(test)]
mod test {
    use super::*;
    use clap::Parser;
    use sassy::profiles::{Iupac, Profile};
    use std::{fs, io::Write};
    use tempfile::NamedTempFile;

    fn mock_guide_txt_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        let g1_txt = format!("TAGCATCAGCTACGNGG\n");
        file.write_all(g1_txt.as_bytes()).unwrap();
        file
    }

    fn mock_target_fasta() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        let t1_fasta = format!(">exact\nTAGCATCAGCTACGAGG\n");
        let t1_pam_mutated_fasta = format!(">pam_mutated\nTAGCATCAGCTACGACG\n"); // C instead of G in PAM
        let t1_n_frac_failure_fasta = format!(">n_frac\nTNNNATCAGCTACGAGG\n"); // 3 N's out of 17 total bases
        file.write_all(t1_fasta.as_bytes()).unwrap();
        file.write_all(t1_pam_mutated_fasta.as_bytes()).unwrap();
        file.write_all(t1_n_frac_failure_fasta.as_bytes()).unwrap();
        file
    }

    #[derive(Default)]
    struct Counts {
        exact: usize,
        pam_mutated: usize,
        n_frac: usize,
    }

    fn get_counts_from_output(tmp_path: &PathBuf) -> Counts {
        let output = fs::read_to_string(tmp_path).unwrap();
        let mut counts = Counts::default();
        for line in output.lines().skip(1) {
            let parts: Vec<&str> = line.split('\t').collect();
            let target = parts[1];
            match target {
                "exact" => counts.exact += 1,
                "pam_mutated" => counts.pam_mutated += 1,
                "n_frac" => counts.n_frac += 1,
                _ => panic!("Unknown target: {target}"),
            }
        }
        counts
    }

    #[test]
    fn test_crispr() {
        let tmp_output = NamedTempFile::new().unwrap();
        let guide_file = mock_guide_txt_file();
        let target_file: NamedTempFile = mock_target_fasta();
        let mut args = CrisprArgs {
            guide: guide_file.path().to_string_lossy().to_string(),
            k: 1,
            output: Some(tmp_output.path().to_path_buf()),
            max_n_frac: 1.0,
            threads: None,
            pam_length: 3,
            allow_pam_edits: true,
            no_rc: false,
            v2: false,
            path: target_file.path().to_path_buf(),
        };
        crispr(&args);
        let found_counts = get_counts_from_output(&tmp_output.path().to_path_buf());
        assert_eq!(found_counts.exact, 2);
        assert_eq!(found_counts.pam_mutated, 1);
        assert_eq!(found_counts.n_frac, 2);

        // If we now disable PAM mutations we should have zero for pam_mutated
        // exact also changes as now it cant move on the last char anymore
        args.allow_pam_edits = false;
        crispr(&args);
        let found_counts = get_counts_from_output(&tmp_output.path().to_path_buf());
        assert_eq!(found_counts.exact, 1);
        assert_eq!(found_counts.pam_mutated, 0);
        assert_eq!(found_counts.n_frac, 1);

        // We have 3 N's / 17 = 0.176..
        let expected_treshold = (3.0 / 17.0) as f32;
        // Slightly above threshold should still pass
        args.max_n_frac = expected_treshold + 0.01;
        crispr(&args);
        let found_counts = get_counts_from_output(&tmp_output.path().to_path_buf());
        assert_eq!(found_counts.n_frac, 1);
        // Setting slightly below the threshold should drop the match
        args.max_n_frac = expected_treshold - 0.01;
        crispr(&args);
        let found_counts = get_counts_from_output(&tmp_output.path().to_path_buf());
        assert_eq!(found_counts.n_frac, 0);

        let run_shared_search = |pam_edits| {
            let mut argv = vec![
                "search",
                "--crispr",
                "--pattern-file",
                &args.guide,
                "-k",
                "1",
                "--max-n-frac",
                "1",
                "--matches",
                tmp_output.path().to_str().unwrap(),
                target_file.path().to_str().unwrap(),
            ];
            if pam_edits {
                argv.push("--allow-pam-edits");
            }
            crate::grep::SearchArgs::try_parse_from(argv).unwrap().run();
            get_counts_from_output(&tmp_output.path().to_path_buf())
        };
        assert_eq!(run_shared_search(false).pam_mutated, 0);
        assert_eq!(run_shared_search(true).pam_mutated, 1);
    }

    #[test]
    fn v2_crispr_maps_reversed_text_hits_back_to_forward_coordinates() {
        let mut guide_file = NamedTempFile::new().unwrap();
        let guide = "TAGCATCAGCTACGNGG";
        let second_guide = "CTTCTTCTTCTTCTNGG";
        writeln!(guide_file, "{guide}\n{second_guide}").unwrap();
        let region = "TAGCATCAGCTACGAGG";
        let reverse_region = Iupac::reverse_complement(region.as_bytes());
        let reverse_region = std::str::from_utf8(&reverse_region).unwrap();
        let mut target_file = NamedTempFile::new().unwrap();
        writeln!(
            target_file,
            ">forward\n{region}\n>reverse\nAAAAA{reverse_region}TT"
        )
        .unwrap();
        let output = NamedTempFile::new().unwrap();

        let mut args = CrisprArgs {
            guide: guide_file.path().to_string_lossy().into_owned(),
            k: 0,
            output: Some(output.path().to_path_buf()),
            max_n_frac: 0.0,
            threads: Some(1),
            pam_length: 3,
            allow_pam_edits: false,
            no_rc: false,
            v2: true,
            path: target_file.path().to_path_buf(),
        };
        crispr(&args);
        let results = fs::read_to_string(output.path()).unwrap();
        let len = region.len();
        let forward = format!("{guide}\tforward\t0\t+\t0\t{len}\t{region}\t{len}=");
        let reverse = format!("{guide}\treverse\t0\t-\t5\t{}\t{region}\t{len}=", 5 + len);
        assert!(results.lines().any(|line| line == forward));
        assert!(results.lines().any(|line| line == reverse));

        args.no_rc = true;
        crispr(&args);
        let results = fs::read_to_string(output.path()).unwrap();
        assert!(results.lines().any(|line| line == forward));
        assert!(results.lines().skip(1).all(|line| !line.contains("\t-\t")));
    }
}
