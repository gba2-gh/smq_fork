# Results and local artifacts

`results/` keeps Markdown, text, PDF and Word reports, together with CSV, TSV, Excel and OpenDocument spreadsheets. Reports over 100 MiB are excluded from this published tree.

Supporting files are stored locally under `large/results/`, with the original subfolder paths preserved. This includes model checkpoints, arrays, predictions, JSON metadata, logs, figures, HTML visualizations and analysis scripts. The `large/` folder is excluded from Git.

For example:

```text
results/week5/example/checkpoint.pt
large/results/week5/example/checkpoint.pt
```

`large-files.csv` records each relocated entry's original path, archive path, byte size and modification timestamp. Its `entry_type` and `link_target` columns also document the relocated dependency junction. The junction's target was not traversed or modified.

All 340 original reports and spreadsheets were retained. The split moved 138,272 supporting files, plus the dependency junction. Empty original directories contain `.gitkeep` files so Git preserves the folder structure.

Some reports link to supporting artifacts now stored in `large/`. To use analysis scripts that require their original paths, restore the artifacts locally from the SMQ repository root:

```powershell
python script/restore_result_artifacts.py
python script/restore_result_artifacts.py --restore
```

The first command only checks availability. The second moves archived entries back without overwriting existing entries. The local `large/` folder is required; it is not included in a Git clone. Supporting file types remain ignored by Git when restored.
