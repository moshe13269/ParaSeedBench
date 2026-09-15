# Human annotation procedure

## Blinding and assignment

Raters A and B receive the same `index.html` and `images/` directory. Each receives
only their own blank CSV. During independent labeling, do not reveal model identity,
automatic labels, the other rater's states, or `PRIVATE_machine_key.csv`.

The audit shuffles neutral IDs such as `item_00000`. Visual style may still reveal
a model family; the protocol does not claim perfect perceptual blinding.

## State encoding

For each image, enter one binary digit per displayed requirement, in displayed
order. `1` means satisfied and `0` means violated. Examples:

- one requirement: `0` or `1`;
- two requirements: `01`;
- four requirements: `0011`.

Treat `human_atom_state` as text. In LibreOffice, select the entire column before
editing and set **Format Cells → Numbers → Text**. This preserves leading zeros.
Do not delete zeros: every atom requires an explicit decision.

Count only visible instances. Use image coordinates for spatial relations. Color
and relation atoms fail if a required object is absent. Ignore unrequested
background objects. Do not infer an object from the prompt. Use `notes` for genuine
ambiguity, but still complete every state before scoring.

## Validation

The scorer rejects missing/extra/duplicate IDs, non-binary states, wrong state
lengths, or partial grids. Run it first without adjudication. It writes
`adjudication_blank.csv` and `adjudication.html` if A and B differ.

Adjudication is a separate resolution pass, not a third independent rater. The
adjudicator sees only disagreement images and requirements; automatic labels must
remain hidden. The file may contain exactly the disagreement IDs or one full
resolved audit.

## Browser after a reboot

On Linux, `py` is a Windows launcher and will not exist. Use:

```bash
cd /path/to/outputs/v2_main_v030_rtx6000ada
python3 -m http.server 8000
```

If the browser is on another machine, use an SSH tunnel rather than exposing the
server publicly:

```bash
ssh -L 8000:localhost:8000 user@workstation
```

Then open `http://localhost:8000/audit/index.html` locally.
