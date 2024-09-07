import React, { useEffect, useState, useMemo} from 'react';
import { Typography, Descriptions } from 'antd';
import { interpolateTurbo } from 'd3-scale-chromatic';
import { rgb } from 'd3-color';
const { asyncBufferFromUrl, parquetRead } = await import('hyparquet')
import { useRouter } from 'next/router'; // Import useRouter from next/router

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

const FeatureDetails = ({ 
  feature,
  model,
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
  useEffect(() => {
    if(!model || !feature) return;
    const asyncRead = async () => {
      const buffer = await asyncBufferFromUrl(`${basePath}/models/${model.label}/samples/chunk_${chunkMapping[feature.feature]}.parquet?cachebust=1`)
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
  }, [feature, chunkMapping, model, basePath])

  useEffect(() => {
    console.log("samples", samples.length)
  }, [samples])

  const featureColor = useMemo(() => interpolateTurbo(feature?.order), [feature])

  return (
    <div className={styles.details}>
      {!feature ? <Paragraph>Select a feature to view details.</Paragraph>: <>
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
                <div className={styles.sampleId}><a href={sample.url} target="_blank">{sample.id}</a></div>
                <div className={styles.sampleText}>{sample.text}</div>
                <div className={styles.sampleTopFeatures}>
                  {sample.top_acts.map((act,i) => {
                    let f = features[sample.top_indices[i]]
                    return {
                      i,
                      feature: f,
                      activation: act,
                      percent: act/f.max_activation
                    }
                  })
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
