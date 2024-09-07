import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { useRouter } from 'next/router'; // Import useRouter from next/router
import Link from 'next/link';
import { Typography, Card, Row, Col, Select } from 'antd';

import { interpolateTurbo, interpolateCool } from 'd3-scale-chromatic';
import { quadtree } from 'd3-quadtree'; // Import quadtree from d3
import { format } from 'd3-format';

import {Tooltip} from 'react-tooltip';

import Layout from '../components/Layout';
import FeatureSelect from '../components/FeatureSelect';
import FeatureDetails from '../components/FeatureDetails';
import Scatter from '../components/Scatter';
import StaticScatter from '../components/StaticScatter';


const { asyncBufferFromUrl, parquetRead } = await import('hyparquet')

const getWindow = () => (typeof window !== 'undefined' ? window : { location: { hash: '' } });

import styles from './index.module.css';

const { Title } = Typography;
const showInt = format(",d")

const models = [
   { label: "NOMIC_FWEDU_25k", value: "NOMIC_FWEDU_25k" },
   { label: "NOMIC_FWEDU_100k (coming soon)", value: "NOMIC_FWEDU_100k", disabled: true },
]

export default function Home() {
  // unfortunately regl-scatter doesn't even render in iOS, and has trouble on Android
  const [isIOSDevice, setIsIOSDevice] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkIsIOS = () => {
      return /iPhone|iPad|iPod/i.test(navigator.userAgent);
    }

    const checkIsMobileDevice = () => {
      return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    };

    setIsIOSDevice(checkIsIOS());
    setIsMobile(checkIsMobileDevice());
  }, []);

  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const mainCardRef = useRef(null);

  const router = useRouter(); // Use useRouter from next/router
  const basePath = useMemo(() => router.basePath, [router])
  const queryParams = router.query;
  // const initialModel = queryParams.model || models[0].value;
  // const initialFeature = queryParams.feature;
  const initialModel = getWindow().location.hash.split('&').find(param => param.startsWith('model='))?.split('=')[1] || models[0].value;
  const initialFeature = getWindow().location.hash.split('&').find(param => param.startsWith('feature='))?.split('=')[1];


  const [selectedModel, setSelectedModel] = useState(models.find(m => m.value === initialModel) || models[0]);
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [features, setFeatures] = useState([])

  useEffect(() => {
    console.log("INITIAL FEATURE", initialFeature, features.length)
    if (initialFeature && features.length) {
      const feature = features.find(f => f.feature === parseInt(initialFeature));
      setSelectedFeature(feature);
      if(feature)
        setSelectedIndices([feature.feature])
    }
  }, [features, initialFeature]);

  useEffect(() => {
    const currentHash = getWindow().location.hash;
    const newHash = currentHash.replace(/model=[^&]*/, `model=${selectedModel?.value}`);
    getWindow().location.hash = newHash
  }, [selectedModel]);


  useEffect(() => {
    const updateDimensions = () => {
      if (mainCardRef.current) {
        const { offsetWidth, offsetHeight } = mainCardRef.current;
        // console.log("DIMENISONS", mainCardRef.current.clientHeight, mainCardRef.current.offsetHeight)
        setDimensions({
          width: offsetWidth - 2,
          height: isMobile ? 300 : offsetHeight - 148, // Subtracting the various headers and pieces
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);

    return () => window.removeEventListener('resize', updateDimensions);
  }, [isMobile]);

  const handleModelSelect = (model) => {
    setSelectedModel(model)
  }

  const [points, setPoints] = useState([])
  useEffect(() => {
    const asyncRead = async () => {
      const buffer = await asyncBufferFromUrl(`${basePath}/models/${selectedModel.label}/features.parquet?cachebust=1`)
      const data = await parquetRead({
        file: buffer,
        onComplete: data => {
          // let pts = []
          console.log("DATA", data)
          let fts = data.map(f => {
            // pts.push([f[2], f[3], parseInt(f[5])])
            return {
              feature: parseInt(f[0]),
              max_activation: f[1],
              x: f[2],
              y: f[3],
              top10_x: f[4],
              top10_y: f[5],
              label: f[6],
              order: f[7],
            }
          })
          // .filter(d => d.label.indexOf("linear") >= 0)
          // .sort((a,b) => a.order - b.order)
          let pts = fts.map(f => [f.top10_x, f.top10_y, f.order])
          setFeatures(fts)
          setPoints(pts)
        }
      })
    }
    asyncRead()
  }, [selectedModel, basePath])

  const [quadtreeInstance, setQuadtreeInstance] = useState(null);

  useEffect(() => {
    if (features.length) {
      const qt = quadtree()
        .x(d => d.top10_x)
        .y(d => d.top10_y)
        .addAll(features);
      setQuadtreeInstance(qt);
    }
  }, [features]);

  const findNearestFeatures = useCallback((feature, count = 5) => {
    if (!quadtreeInstance) return [];
    const nearest = [];
    const searchRadius = 1
    quadtreeInstance.visit((node, x0, y0, x1, y1) => {
      if (!node.length) {
        do {
          const d = node.data;
          const dx = d.top10_x - feature.top10_x;
          const dy = d.top10_y - feature.top10_y;
          const distance = Math.sqrt(dx * dx + dy * dy); // Calculate distance without modifying node
          nearest.push({ ...d, distance });
        } while (node = node.next);
      }
      return x0 > feature.top10_x + searchRadius || x1 < feature.top10_x - searchRadius || y0 > feature.top10_y + searchRadius || y1 < feature.top10_y - searchRadius;
    });
    return nearest.sort((a, b) => a.distance - b.distance).slice(0, count);
  }, [quadtreeInstance, features])

  const [nearestFeatures, setNearestFeatures] = useState([])
  useEffect(() => {
    if (selectedFeature) {
      const nearestFeatures = findNearestFeatures(selectedFeature, 6);
      setNearestFeatures(nearestFeatures.slice(1))
    }
  }, [selectedFeature, quadtreeInstance]);

  const [modelMetadata, setModelMetadata] = useState(null)
  const [chunkMapping, setChunkMapping] = useState(null)
  useEffect(() => {
    const asyncRead = async () => {
      const meta = await fetch(`${basePath}/models/${selectedModel.label}/metadata.json?cachebust=1`).then(r => r.json())
      setModelMetadata(meta)
      const chunkMapping = await fetch(`${basePath}/models/${selectedModel.label}/chunk_mapping.json?cachebust=1`).then(r => r.json())
      setChunkMapping(chunkMapping)
    }
    asyncRead()
  }, [selectedModel, basePath])

  // ====================================================================================================
  // Scatterplot related logic
  // ====================================================================================================
  // this is a reference to the regl scatterplot instance
  // so we can do stuff like clear selections without re-rendering
  const [scatter, setScatter] = useState(null)
  const [xDomain, setXDomain] = useState([-1, 1]);
  const [yDomain, setYDomain] = useState([-1, 1]);
  const handleView = useCallback((xDomain, yDomain) => {
    setXDomain(xDomain);
    setYDomain(yDomain);
  }, [setXDomain, setYDomain])
  // Selection via Scatterplot
  // indices of items selected by the scatter plot
  const [selectedIndices, setSelectedIndices] = useState([]);


  // Hover via scatterplot or tables
  // index of item being hovered over
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const [hoveredFeature, setHoveredFeature] = useState(null);
  // const [tooltipContent, setTooltipContent] = useState('');

  useEffect(() => {
    if (hoveredIndex !== null && hoveredIndex !== undefined) {
      setHovered(hoveredIndex);
      const feature = features[hoveredIndex];
      // console.log("hovered", hoveredIndex, feature)
      if (feature && xDomain && yDomain) {
        const xPos = ((feature.top10_x - xDomain[0]) / (xDomain[1] - xDomain[0])) * dimensions.width;
        const yPos = ((feature.top10_y - yDomain[1]) / (yDomain[0] - yDomain[1])) * (dimensions.height) + 118;
        setTooltipPosition({ 
          x: xPos,// - .5*16, 
          y: yPos// - .67*16 
        });
        setHoveredFeature(feature)
      }
    } else {
      setHovered(null);
      setHoveredFeature(null)
    }
  }, [hoveredIndex, setHovered, features, xDomain, yDomain, dimensions]);

  const handleHover = useCallback((index) => {
    setHoveredIndex(index);
  }, [setHoveredIndex])


  const handleSelected = useCallback((indices) => {
    console.log("handle selected", indices, features[indices[0]])
    setSelectedFeature(features[indices[0]])
    setSelectedIndices(indices)
    // router.push({
    //   pathname: router.pathname,
    //   query: { ...router.query, feature: indices[0] }
    // });
    getWindow().location.hash = `model=${selectedModel.value}&feature=${indices[0] || ""}`;

  }, [setSelectedIndices, features])

  const handleFeatureSelect = useCallback((feature) => {
    console.log("FEATURE SELECTED", feature)
    setSelectedFeature(feature);
    if(feature) {
      scatter?.select([feature.feature])
      getWindow().location.hash = `model=${selectedModel.value}&feature=${feature.feature}`;

    } else {
      scatter?.select([])
      getWindow().location.hash = `model=${selectedModel.value}&feature=`;
    }
  }, [scatter, setSelectedIndices, selectedModel])

  const handleFeatureHover = useCallback((feature) => {
    setHoveredIndex(feature?.feature)
  }, [setHoveredIndex])

  const [filteredIndices, setFilteredIndices] = useState(null)
  const handleFilter = (options) => {
    const indices = options.slice(0,100).map(o => o.feature)
    setFilteredIndices(indices)
  }


  return (
    <Layout>
      <div className={styles.homeContainer}>
        {/* <Title className={styles.pageTitle}>Latent Taxonomy</Title> */}
        {/* <Row className={styles.fullHeightRow} gutter={[24, 24]}> */}
        <Row className={`${styles.fullHeightRow} ${isMobile ? styles.mobileRow : ''}`} gutter={[24, 24]}>
          <Col xs={24} lg={12} className={styles.fullHeightCol} >
            <Card title={
              <div className={styles.modelTitle}>
                <div className={styles.modelMetadata}>
                  <Select 
                    style={{ width: '280px' }} 
                    options={models} 
                    value={models[0]} 
                    onChange={handleModelSelect} 
                    data-tooltip-id="modelTooltip" 
                  />
                  {modelMetadata && <span>{showInt(modelMetadata.num_latents)} features</span>}
                </div>
                {modelMetadata && <span><Link href="/articles/about">more info</Link></span>}
                <Tooltip id="modelTooltip">
                  Select a model
                </Tooltip>
              </div>
              } 
            className={styles.fullHeightCard} 
            ref={mainCardRef}>
              <div className={styles.extraInfo}>
                    { modelMetadata && !isMobile && 
                    <p>Each dot is a feature of an <a href={`https://huggingface.co/${modelMetadata.repo}`} target='_blank'>SAE</a> 
                      <span> trained on 100 billion tokens of <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu" target='_blank'>FineWeb-edu </a></span>
                    <br></br>
                      embedded via <a href={`https://huggingface.co/${modelMetadata.source_model}`} target='_blank'>
                      {modelMetadata.source_model}
                      </a>
                    </p> }
                    {isMobile && <p>
                      {modelMetadata && <span>{showInt(modelMetadata.num_latents)} features<br></br></span>}
                      <Link href="/articles/about">more info</Link>
                    </p>}

              </div>
                <div className={styles.scatters} style={{ width: dimensions.width, height: dimensions.height }}>
                  {points.length ? <>
                  { !isMobile ? <Scatter
                    points={points}
                    duration={2000}
                    width={dimensions.width}
                    height={dimensions.height}
                    colorScaleType="continuous"
                    // colorScaleType="categorical"
                    colorInterpolator={interpolateTurbo}
                    pointScale={1}
                    pointColor={"#444"}
                    onScatter={setScatter}
                    onView={handleView}
                    onSelect={handleSelected}
                    onHover={handleHover}
                  /> : <StaticScatter
                    points={points}
                    fill="gray"
                    opacity={0.25}
                    size="4"
                    xDomain={xDomain}
                    yDomain={yDomain}
                  width={dimensions.width}
                  height={dimensions.height}
                /> }

                {filteredIndices?.length ? <StaticScatter
                  points={filteredIndices.map(i => points[i])}
                  stroke="black"
                  fill="gray"
                  size="8"
                  xDomain={xDomain}
                  yDomain={yDomain}
                  width={dimensions.width}
                  height={dimensions.height}
                /> : null}

                {selectedIndices?.length ? <StaticScatter
                  points={selectedIndices.map(i => points[i])}
                  stroke="black"
                  fill="black"
                  // symbol="⭐️"
                  size="12"
                  xDomain={xDomain}
                  yDomain={yDomain}
                  width={dimensions.width}
                  height={dimensions.height}
                /> : null }
              </> : null }
            </div>
            
            <div
              data-tooltip-id="featureTooltip"
              style={{
                position: 'absolute',
                left: tooltipPosition.x,
                top: tooltipPosition.y,
                pointerEvents: 'none',
              }}
            ></div>
            <Tooltip id="featureTooltip" 
              isOpen={hoveredIndex !== null}
              delayShow={0}
              delayHide={0}
              delayUpdate={0}
              style={{
                position: 'absolute',
                left: tooltipPosition.x,
                top: tooltipPosition.y,
                pointerEvents: 'none',
              }}
            >
              {hoveredFeature && <span>{hoveredFeature.feature}: {hoveredFeature.label}</span>}
            </Tooltip>
            </Card>
          </Col>
          <Col xs={24} lg={12} className={styles.fullHeightCol}>
          <Card 
              title={
                <FeatureSelect 
                  options={features} 
                  value={selectedFeature} 
                  onSelect={handleFeatureSelect} 
                  onFilter={handleFilter}
                />
              } 
              className={styles.fullHeightCard}
            >
             <FeatureDetails 
              feature={selectedFeature} 
              model={selectedModel} 
              chunkMapping={chunkMapping} 
              nearestFeatures={nearestFeatures}
              features={features}
              onHover={handleFeatureHover}
              onSelect={handleFeatureSelect} 
            /> 
            </Card>
          </Col>
        </Row>

      </div>
    </Layout>
  );
}