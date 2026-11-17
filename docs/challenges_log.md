2026-11-16 (Shreya): CNN merged to main (src/models/cnn.py, tests/test_cnn.py, 9 tests pass). Width solved to about 75k parameters (75,815 for kernel 5 with 3 blocks; all configs 74.9k to 76.2k, see docs/cnn_param_counts.txt). Tuning on UCI is waiting on Sanjana's harness and splits/uci.json.


2026-11-15 (Sharanya): PAMAP2 subject 109 has zero windows of the 5 classes in both Protocol and Optional (readme: hardware problems left some activities of some subjects partly or completely missing). scripts/make_splits.py --real asserted 9 subjects for pamap2_hand. Resolved 2026-11-16: team agreed Option A for 8 subjects (see docs/decision_log.md).
