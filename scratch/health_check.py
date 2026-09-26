"""Read-only project health check: verifies locked artifacts are consistent and loadable."""
import json
import sys

sys.path.append('src')

from model_artifacts import verify_and_load_locked_model

model, feats, thr = verify_and_load_locked_model(
    'models/xgboost_entity_resolution_phase5_configA.json',
    'models/phase5_configA_feature_schema.json',
    'models/phase5_configA_threshold.json',
)
print('model loaded OK, n_features =', model.n_features_in_)
print('schema features =', len(feats))
print('locked threshold =', thr)

meta = json.load(open('models/phase5_configA_metadata.json', encoding='utf-8'))
schema = json.load(open('models/phase5_configA_feature_schema.json', encoding='utf-8'))
thresh = json.load(open('models/phase5_configA_threshold.json', encoding='utf-8'))

print('meta best_iteration =', meta['best_iteration'],
      '| val macroF0.5 =', round(meta['validation_macro_f0_5'], 4))
print('fingerprint alignment (meta/schema/thresh):',
      meta['config_fingerprint'] == schema['config_fingerprint'] == thresh['config_fingerprint'])
print('schema feature order matches metadata schema:',
      schema['feature_names'] == meta['feature_schema'])
print('has name_tfidf_cosine feature:', 'name_tfidf_cosine' in feats)
