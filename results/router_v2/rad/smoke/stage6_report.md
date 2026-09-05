# Stage 6 RAD V2 Validation Report

Status: **PASS**

- Best fixed lambda selected on validation: `0.5`
- Best heuristic selected on validation: `heuristic_js`
- TARO validation mean lambda: `0.9002140829106793`
- V2 history validation mean lambda: `0.9655981808900833`
- Combined held-out records: `72`
- Protected artifacts unchanged: `True`
- Router collapse absent: `True`

Primary Pareto uses sentiment alignment versus PPL degradation relative to the base method from the same model family. Raw V1 and V2 PPL values must not be compared as if they used the same backbone/protocol.
