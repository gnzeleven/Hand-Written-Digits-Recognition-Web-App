import React, { useState, useRef, useEffect } from "react";
import { SketchField, Tools } from "react-sketch";
import { Button } from "react-bootstrap";
import axios from "axios";
import GridComponent from "./GridComponent";

const styles = {
  draw: {
    margin: "0 auto",
  },
};

const endPoint = "http://localhost:8000/api/digits/";

const Draw = () => {
  // lineColor - to handle erase and redraw
  // success - to show the results
  const [lineColor, setLineColor] = useState("white");
  const [success, setSuccess] = useState(false);

  // results - top two predictions
  // scores - corresponding scores
  const [results, setResults] = useState([]);
  const [scores, setScores] = useState([]);

  // useRef returns a mutable ref object whose .current property
  // is initialized to the passed argument (initialValue). The returned
  // object will persist for the full lifetime of the component.
  const sketch = useRef();

  // handles onClick event for save
  const handleSave = () => {
    const canvas = sketch.current.toDataURL();
    // console.log(canvas);
    //saveAs(canvas, "digit.jpg");
    sendData(canvas);
  };

  // handles onClick event for reset
  const handleReset = () => {
    sketch.current.clear();
    sketch.current._backgroundColor("black");
    setResults([]);
    setScores([]);
    setSuccess(false);
  };

  // uses Axios to send post request to the DRF backend
  const sendData = (canvas) => {
    const headers = {
      accept: "application/json",
    };

    const fd = new FormData();
    fd.append("image", canvas);

    axios
      .post(endPoint, fd, { headers: headers })
      .then((res) => {
        //console.log(res.data);
        getImageResult(res.data.id);
      })
      .catch((error) => {
        console.log(error);
      });
  };

  // uses Axios to send a get request with the id to DRF backend
  const getImageResult = (id) => {
    axios.get(`${endPoint}${id}`).then((res) => {
      setResults([res.data.result1, res.data.result2]);
      setScores([res.data.score1, res.data.score2]);
      setSuccess(true);
    });
  };

  return (
    <React.Fragment>
      {/* Display GridComponent with props */}
      <GridComponent results={results} />
      {/* Show the results depending on the success variable */}
      {success && (
        <h5 className="result">
          {" "}
          Best guess: {results[0]}; score: {scores[0]}{" "}
        </h5>
      )}
      {success && (
        <p className="result">
          {" "}
          Next Best guess: {results[1]}; score: {scores[1]}{" "}
        </p>
      )}
      {/* SketchField component from react-sketch library
      https://www.npmjs.com/package/react-sketch */}
      <SketchField
        width="450px"
        height="450px"
        ref={sketch}
        style={styles.draw}
        tool={Tools.pencil}
        backgroundColor="black"
        lineColor={lineColor}
        imageFormat="jpg"
        lineWidth={35}
      />
      {/* Buttons and their functionalities */}
      <div className="mt-3">
        <Button variant="primary" onClick={handleSave}>
          Save
        </Button>
        <Button variant="secondary" onClick={handleReset}>
          Reset
        </Button>
        <Button
          variant="info"
          onClick={() => {
            setLineColor("white");
          }}
        >
          Draw
        </Button>
        <Button
          variant="dark"
          onClick={() => {
            setLineColor("black");
          }}
        >
          Erase
        </Button>
      </div>
    </React.Fragment>
  );
};

export default Draw;
