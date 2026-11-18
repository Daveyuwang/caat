BEGIN TRANSACTION;
CREATE TABLE audit (id INTEGER PRIMARY KEY CHECK(id=1), binding TEXT, state TEXT);
INSERT INTO "audit" VALUES(1,'{"accepted": 100, "accepted_population": 580, "accepted_population_sha256": "4f8c708d4c928256610bb3ff87627566527a1c74e3808a8b55390b6d5df65ccf", "block_size": 10, "graders": ["anli"], "holdout_confidence": 0.95, "label_budget": 100, "risk_target": 0.2, "sample_seed": 42, "selection": null, "threshold": 4.75, "threshold_report_sha256": "5624e547f5a68e1cb0955a8c5c0f363e769a2d71ab99875314107cd9168bdbc8", "total": 879}','{"requests": [{"id": "8ccb39fdddad34e5eb45-1", "start": 0, "end": 10, "graders": ["anli"]}], "grades": {"anli": {"f80b1c59-f5fe-4467-87a5-06924f369ce5": false}}}');
COMMIT;
