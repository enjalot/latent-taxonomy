import React, { useEffect, useState, useMemo} from 'react';
import { Typography, Descriptions } from 'antd';
import { interpolateTurbo } from 'd3-scale-chromatic';
import { rgb } from 'd3-color';
const { parquetRead } = await import('hyparquet')
import { useRouter } from 'next/router'; // Import useRouter from next/router

// Fetch the whole file in a single GET and wrap it as an in-memory AsyncBuffer.
// hyparquet's asyncBufferFromUrl reads the footer via HTTP Range requests, which
// breaks on GitHub Pages when the parquet is served gzip-compressed (the range
// overruns the gzipped body -> 416 -> "footer != PAR1"). A full GET lets the
// browser transparently decompress gzip.
const bufferFromUrl = async (url) => {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`)
  const arrayBuffer = await res.arrayBuffer()
  return { byteLength: arrayBuffer.byteLength, slice: (start, end) => arrayBuffer.slice(start, end) }
}

import styles from './FeatureDetails.module.css';

function yiq(color) {
  const {r, g, b} = rgb(color);
  return (r * 299 + g * 587 + b * 114) / 1000 / 255; // returns values between 0 and 1
}

const { Title, Paragraph } = Typography;

const ActivationBar = ({
  feature,
  activation,
  content,
  onHover = () => {},
  onSelect = () => {},
}) => {
  const featureColor = useMemo(() => interpolateTurbo(feature?.order), [feature])
  return (
    <div className={styles.sampleActivationBar} 
      onMouseEnter={() => onHover(feature)}
      onMouseLeave={() => onHover(null)}
      onClick={() => onSelect(feature)}
    >
      <div className={styles.sampleActivationBarForeground} 
        style={{
          width: `${activation/feature.max_activation * 100}%`, 
          backgroundColor: featureColor,
        }}
      >
      </div>
      <div className={styles.sampleActivationBarLabel} 
      style={{
        // color: yiq(featureColor) >= 0.6 ? "#111" : "white",
      }}>
        <span>{feature.feature}: {feature.label}</span><span>{activation.toFixed(3)} ({(100*activation/feature.max_activation).toFixed(0)}%)</span>
      </div>
    </div>
  )
}

// Article-style scorecard for a model's eval numbers (metadata.evals).
// Shown when no feature is selected; hidden entirely for models without evals.
const showPct = (v) => (v === null || v === undefined) ? "—" : `${(v * 100).toFixed(1)}%`;
const EvalsCard = ({ metadata }) => {
  if (!metadata?.evals) return null;
  const e = metadata.evals;
  const rows = [
    ["Reconstruction (FVU, held-out)", e.fvu != null ? e.fvu.toFixed(3) : "—"],
    ["Dead features", e.dead_pct != null ? `${e.dead_pct}%` : "—"],
    ["Coherence (stratified)", e.coherence_stratified != null ? showPct(e.coherence_stratified) : "not yet run"],
  ];
  if (e.ndcg_recovered) {
    Object.entries(e.ndcg_recovered).forEach(([ds, v]) => {
      rows.push([`nDCG@10 recovered — ${ds.replace(/_/g, "-")}`, showPct(v)]);
    });
  }
  return (
    <div className={styles.evalsCard}>
      <h2>Model scorecard</h2>
      {metadata.corpus && <p className={styles.evalsCorpus}>Trained on {metadata.corpus}.</p>}
      {metadata.matryoshka_levels && (
        <p className={styles.evalsCorpus}>
          Matryoshka levels {metadata.matryoshka_levels.join(" / ")} with k = {(metadata.ks || []).join(" / ")}.
        </p>
      )}
      <table className={styles.evalsTable}>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}><td>{k}</td><td>{v}</td></tr>
          ))}
        </tbody>
      </table>
      {e.notes && <p className={styles.evalsNotes}>{e.notes}</p>}
    </div>
  );
};

const FeatureDetails = ({
  feature,
  model,
  metadata,
  chunkMapping,
  nearestFeatures,
  features,
  onHover = () => {},
  onSelect = () => {}
}) => {
  // Fetch samples
  const [samples, setSamples] = useState([])
  const router = useRouter(); // Use useRouter from next/router
  const basePath = useMemo(() => router.basePath, [router])

  // look up features by their id (feature ids are NOT array indices when a
  // model publishes a labeled subset of its latents)
  const featureById = useMemo(() => {
    const m = new Map();
    (features || []).forEach(f => m.set(f.feature, f));
    return m;
  }, [features]);

  // Samples can be hosted off-origin (big models keep their ~1GB of sample
  // shards on e.g. a HF dataset — metadata.samples_base_url). During local
  // dev, samples_base_url_local (if present) wins so not-yet-uploaded shards
  // can be served from disk (see pipeline/serve_samples.py).
  const samplesBase = useMemo(() => {
    if (!model) return null;
    const host = typeof window !== 'undefined' ? window.location.hostname : '';
    const isLocal = host === 'localhost' || host === '127.0.0.1';
    if (isLocal && metadata?.samples_base_url_local) return metadata.samples_base_url_local;
    if (metadata?.samples_base_url) return metadata.samples_base_url;
    return `${basePath}/models/${model.value}/samples`;
  }, [model, metadata, basePath]);

  useEffect(() => {
    if(!model || !feature || !chunkMapping || !samplesBase) return;
    const asyncRead = async () => {
      const buffer = await bufferFromUrl(`${samplesBase}/chunk_${chunkMapping[feature.feature]}.parquet?cachebust=1`)
      const data = await parquetRead({
        file: buffer,
        rowFormat: 'object',
        onComplete: data => {
          // console.log("SAMPLE DATA", data)
          let ss = data.map(d => {
            return {
              ...d,
              feature: parseInt(d.feature)
            }
          })
          .filter(d => d.feature === feature.feature)
          console.log("SAMPLES", ss)
          setSamples(ss)
        }
      })
    }
    asyncRead()
  }, [feature, chunkMapping, model, samplesBase])

  useEffect(() => {
    console.log("samples", samples.length)
  }, [samples])

  const featureColor = useMemo(() => interpolateTurbo(feature?.order), [feature])

  return (
    <div className={styles.details}>
      {!feature ? <>
        <Paragraph>Select a feature to view details.</Paragraph>
        <EvalsCard metadata={metadata} />
      </> : <>
        <Title level={4}>{feature.feature}: {feature.label}</Title>

        <div className={styles.similarFeatures}>
          <h2>Similar features</h2>
          <div className={styles.similarFeaturesList}>
            {nearestFeatures.map(f => (
              <div key={"similar-"+f.feature} className={styles.similarFeature}
              onMouseEnter={() => onHover(f)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onSelect(f)}
              >
                <span
                  style={{
                    display: "block",
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    backgroundColor: interpolateTurbo(f.order),
                    marginRight: '8px',
                  }}
                />
                <span style={{ width: 'calc(100% - 20px)' }}>{f.feature}: {f.label}</span>
              </div>
            ))}
          </div>
        </div>

        { samples.length && 
        <div className={styles.samples}>
          <h2>Top activating samples</h2>
          <div>
            {samples.map((sample,i) => (
              <div key={"sample-"+i} className={styles.sample}>
                {/* <ActivationBar
                  feature={feature}
                  activation={sample.activation}
                /> */}
                <div className={styles.sampleId}>{sample.url
                  ? <a href={sample.url} target="_blank">{sample.id}</a>
                  : <span>{sample.id}</span>}</div>
                <div className={styles.sampleText}>{sample.text}</div>
                <div className={styles.sampleTopFeatures}>
                  {(sample.top_acts || []).map((act,i) => {
                    // top_indices hold feature ids; look features up by id
                    // (ids are not array indices for subset-published models)
                    let f = featureById.get(parseInt(sample.top_indices[i]))
                    return {
                      i,
                      feature: f,
                      activation: act,
                      percent: f ? act/f.max_activation : 0
                    }
                  })
                  .filter(f => f.feature) // unlabeled features have no entry
                  //.sort((a,b) => b.percent - a.percent)
                  .slice(0, 10)
                  .map(f => (
                    <ActivationBar
                      key={f.i}
                      feature={f.feature}
                      activation={f.activation}
                      onHover={onHover}
                      onSelect={onSelect}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>}
        
      </>}
    </div>
  );
};

export default FeatureDetails;
