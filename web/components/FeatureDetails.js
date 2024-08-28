import React, { useEffect, useState, useMemo} from 'react';
import { Typography, Descriptions } from 'antd';
import { interpolateTurbo } from 'd3-scale-chromatic';
import { rgb } from 'd3-color';
const { asyncBufferFromUrl, parquetRead } = await import('hyparquet')

import styles from './FeatureDetails.module.css';

function yiq(color) {
  const {r, g, b} = rgb(color);
  return (r * 299 + g * 587 + b * 114) / 1000 / 255; // returns values between 0 and 1
}

const { Title, Paragraph } = Typography;

const FeatureDetails = ({ 
  feature,
  model,
  chunkMapping,
  nearestFeatures,
  onHover = () => {},
  onSelect = () => {}
}) => {
  // Fetch samples 
  const [samples, setSamples] = useState([])
  useEffect(() => {
    if(!model || !feature) return;
    const asyncRead = async () => {
      const buffer = await asyncBufferFromUrl(`/models/${model.label}/samples/chunk_${chunkMapping[feature.feature]}.parquet`)
      const data = await parquetRead({
        file: buffer,
        onComplete: data => {
          // console.log("SAMPLE DATA", data)
          let ss = data.map(f => {
            return {
              id: f[0],
              text: f[1],
              feature: parseInt(f[2]),
              activation: f[3],
              top_acts: f[4],
              top_indices: f[5],
            }
          }).filter(d => d.feature === feature.feature)
          console.log("SAMPLES", ss)
          setSamples(ss)
        }
      })
    }
    asyncRead()
  }, [feature, chunkMapping, model])

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
                <div className={styles.sampleActivationBar} style={{
                  width: `100%`, 
                  border: "1px solid lightgray",
                  height: "14px"
                }}>
                  <div style={{
                    width: `${sample.activation/feature.max_activation * 100}%`, 
                    backgroundColor: featureColor,
                    color: yiq(featureColor) >= 0.6 ? "#111" : "white",
                    height: "12px",
                    fontSize: "10px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "left",
                    paddingLeft: "2px"
                  }}>
                    {sample.activation.toFixed(2)}
                  </div>
                </div>
                <div className={styles.sampleText}>{sample.text}</div>
                <div className={styles.sampleTopFeatures}></div>
              </div>
            ))}
          </div>
        </div>}
        
      </>}
    </div>
  );
};

export default FeatureDetails;
