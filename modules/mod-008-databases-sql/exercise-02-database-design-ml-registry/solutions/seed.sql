-- Minimal seed data for the model registry schema. Run after schema.sql.

INSERT INTO experiments (name, description, owner) VALUES
    ('fraud-detection', 'Real-time payment fraud detection', 'team-payments'),
    ('rec-personalization', 'Homepage recommendation reranker', 'team-discovery')
ON CONFLICT (name) DO NOTHING;

INSERT INTO models (experiment_id, name) VALUES
    ((SELECT id FROM experiments WHERE name = 'fraud-detection'),     'fraud-detector'),
    ((SELECT id FROM experiments WHERE name = 'rec-personalization'), 'home-reranker')
ON CONFLICT (name) DO NOTHING;

INSERT INTO model_versions (model_id, version_tag, artifact_uri, framework, hyperparameters, metrics, trained_by) VALUES
    ((SELECT id FROM models WHERE name = 'fraud-detector'), 'v1.0.0',
     's3://models/fraud/v1.0.0/model.pkl', 'sklearn',
     '{"learning_rate":0.05,"n_estimators":300}',
     '{"roc_auc":0.86,"precision":0.78,"recall":0.71}', 'alice'),
    ((SELECT id FROM models WHERE name = 'fraud-detector'), 'v1.1.0',
     's3://models/fraud/v1.1.0/model.pkl', 'sklearn',
     '{"learning_rate":0.03,"n_estimators":500}',
     '{"roc_auc":0.91,"precision":0.82,"recall":0.75}', 'alice'),
    ((SELECT id FROM models WHERE name = 'home-reranker'), 'v0.9.0',
     's3://models/reranker/v0.9.0/model.pt', 'pytorch',
     '{"embedding_dim":128,"dropout":0.2}',
     '{"ndcg@10":0.42,"recall@10":0.55}', 'bob');

INSERT INTO model_version_lineage (child_id, parent_id, edge_type) VALUES
    ((SELECT id FROM model_versions WHERE version_tag = 'v1.1.0'
      AND model_id = (SELECT id FROM models WHERE name = 'fraud-detector')),
     (SELECT id FROM model_versions WHERE version_tag = 'v1.0.0'
      AND model_id = (SELECT id FROM models WHERE name = 'fraud-detector')),
     'retrain');

INSERT INTO deployments (model_version_id, environment, status, endpoint_url) VALUES
    ((SELECT id FROM model_versions WHERE version_tag = 'v1.1.0'
      AND model_id = (SELECT id FROM models WHERE name = 'fraud-detector')),
     'prod', 'active', 'https://model-api/v1.1.0');
