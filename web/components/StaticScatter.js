import React, { useEffect, useRef } from 'react';
import { scaleLinear } from 'd3-scale';

import styles from  "./StaticScatter.module.css"

const StaticScatter = ({ 
  points, 
  fill,
  stroke,
  opacity = 0.75,
  size,
  symbol,
  xDomain, 
  yDomain, 
  width, 
  height
}) => {
  const container = useRef();
  
  useEffect(() => {
    if(xDomain && yDomain) {
      const xScale = scaleLinear()
        .domain(xDomain)
        .range([0, width])
      const yScale = scaleLinear()
        .domain(yDomain)
        .range([height, 0])

      const zScale = (t) => t/(.1 + xDomain[1] - xDomain[0])
      const canvas = container.current
      const ctx = canvas.getContext('2d')
      let rw = zScale(size)
      ctx.clearRect(0, 0, width, height)
      ctx.fillStyle = fill 
      ctx.strokeStyle = stroke
      ctx.font = `${rw}px monospace`
      ctx.globalAlpha = opacity
      // ctx.globalAlpha = 1
      if(!points.length) return
      points.map(point => {
        if(!point) return;
        if(fill)
          // ctx.fillRect(xScale(point[0]) - rw/2, yScale(point[1]) - rw/2, rw, rw);
          ctx.beginPath();
          ctx.arc(xScale(point[0]), yScale(point[1]), rw / 2, 0, 2 * Math.PI);
          ctx.fill();
        if(stroke)
          ctx.stroke();
          // ctx.strokeRect(xScale(point[0]) - rw/2, yScale(point[1]) - rw/2, rw, rw)
        if(symbol)
          ctx.fillText(symbol, xScale(point[0]) - rw/2.2, yScale(point[1]) + rw/3.2)
      })
    }

  }, [points, fill, stroke, size, xDomain, yDomain, width, height])

  return <canvas 
    className={styles.statics}
    ref={container} 
    width={width} 
    height={height} />
};

export default StaticScatter;
