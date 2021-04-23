import React, { useState, useEffect } from "react";
import Box from "@material-ui/core/Box";

// Numbers grid component
const GridComponent = ({ results }) => {
  // Get result1 and result2 from the props
  const result1 = results[0];
  const result2 = results[1];
  // color - set the colors for the grids
  const [color, setColor] = useState({
    0: "grey.300",
    1: "grey.300",
    2: "grey.300",
    3: "grey.300",
    4: "grey.300",
    5: "grey.300",
    6: "grey.300",
    7: "grey.300",
    8: "grey.300",
    9: "grey.300",
  });
  // tempColor - to update colors when the results are there
  let tempColor = {
    0: "grey.300",
    1: "grey.300",
    2: "grey.300",
    3: "grey.300",
    4: "grey.300",
    5: "grey.300",
    6: "grey.300",
    7: "grey.300",
    8: "grey.300",
    9: "grey.300",
  };

  // Effect hook update the color when there's a change in the
  // state of results variable and rerender
  useEffect(() => {
    //console.log(result1);
    tempColor[result1] = "#1bf507";
    tempColor[result2] = "#fcf56a";
    setColor(tempColor);
    //console.log(color);
  }, [results]);

  return (
    <div
      className="gridItems"
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Box display="flex" flexWrap="nowrap">
        <Box p={2} bgcolor={color[0]}>
          0
        </Box>
        <Box p={2} bgcolor={color[1]}>
          1
        </Box>
        <Box p={2} bgcolor={color[2]}>
          2
        </Box>
        <Box p={2} bgcolor={color[3]}>
          3
        </Box>
        <Box p={2} bgcolor={color[4]}>
          4
        </Box>
        <Box p={2} bgcolor={color[5]}>
          5
        </Box>
        <Box p={2} bgcolor={color[6]}>
          6
        </Box>
        <Box p={2} bgcolor={color[7]}>
          7
        </Box>
        <Box p={2} bgcolor={color[8]}>
          8
        </Box>
        <Box p={2} bgcolor={color[9]}>
          9
        </Box>
      </Box>
    </div>
  );
};

export default GridComponent;
