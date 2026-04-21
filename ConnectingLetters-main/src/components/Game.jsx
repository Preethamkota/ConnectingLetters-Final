import React, { useRef, useState, useEffect } from "react";
import confetti from "canvas-confetti";
import { useSearchParams, useNavigate } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import GameCompleted from "./Modals/GameCompleted";
import NextLevel from "./Modals/NextLevel";
import { Application, Assets, Sprite, Text, TextStyle, Graphics } from "pixi.js";
import { Popover } from "bootstrap";
import instructions from "../assets/instructions.wav";
import axios from "axios";
import { startCapture,endCapture,assignInterval,assignUploadUrl,CaptureElements } from "../hooks/useCapture";
export default function Game() {
  const [searchParams] = useSearchParams();
  const appRef = useRef();
  const startTimeRef = useRef(null);
  const triesRef = useRef(0);
  const navigate = useNavigate();
  const stack = [];
  const [tries, setTries] = useState(0);
  const [correct, setCorrect] = useState(0);
  const [lvl, setLvl] = useState(parseInt(searchParams.get("lvl")) ?? 1);
  const [item, setItem] = useState(1);
  const [gameOver, setGameOver] = useState(false);
  const data = useRef();
  const circles = useRef([]);
  

  async function handleNext() {
    triesRef.current += tries;
    setTries(0);
    setCorrect(0);
    if (item < 9) setItem((item) => item + 1);
    else if (lvl < 2) {
      navigate(`/game?lvl=${lvl + 1}`);
      navigate(0);
      // setLvl((lvl) => lvl + 1);
      // setItem(1);
    } else {
      confetti();
      setGameOver(true);
      endCapture();
      const gameId = searchParams.get("gameId");
      const therapistId = localStorage.getItem("therapistId");
      const childId = localStorage.getItem("childId");
      try {
        var c = await axios.post(`https://totalapi.joywithlearning.com/api/data/submitGameDetails/${gameId}/${childId}`, {
          tries: triesRef.current + tries,
          timer: (new Date() - startTimeRef.current) / 1000,
          status: "completed",
          therapistId: therapistId,
          datePlayed : new Date()
        });
      } catch (error) {
        console.error("Error submitting game details:", error);
      }
      console.log(c);
    }
  }

  function handleRestart() {
    setLvl(1);
    setItem(1);
    setGameOver(false);
     // End current capture and start a new capture for the restart
    endCapture();
    navigate("/game?lvl=1")
    startCapture();
  }

  const instruction = new Audio(instructions);
  const getVoice = (lang = "en-US") => {
    const voices = window.speechSynthesis.getVoices();
    return voices.find((voice) => voice.lang === lang) || voices[1];
  };

  function speak(letter, rate = 1, pitch = 1) {
    var msg = new SpeechSynthesisUtterance(letter);
    msg.voice = getVoice();
    msg.rate = rate;
    // msg.pitch = pitch;
    window.speechSynthesis.speak(msg);
  }

  function handleclick(Circle) {
    console.log(Circle.word, Circle.letter);
    stack.push(Circle);
    const letters = Object.keys(data.current[Circle.word])[Circle.letter];
    letters.split("").forEach((letter) => speak(letter));
    if (stack.length === 1) {
      if (Circle.letter !== 0) {
        Circle.tint = "FF0000";
        stack.pop();
        resetColor(Circle);
        setTries((tries) => tries + 1);
        return false;
      }
      Circle.tint = "#FFFF00";
    } else if (
      Circle.letter === stack[stack.length - 2].letter + 1 &&
      stack[stack.length - 2].word === Circle.word
    ) {
      if (stack.length === Object.keys(data.current[Circle.word]).length) {
        window.speechSynthesis.getVoices();
        confetti({
          particleCount: 300,
          spread: 90,
          decay: 0.95,
          scalar: 1.5,
          ticks: 150,
          origin: {
            y: 0.9
          }
        });
        speak(Object.keys(data.current[Circle.word]).join(""));
        while (stack.length !== 0) {
          const elem = stack.pop();
          elem.interactive = false;
          elem.tint = "#00FF00";
        }
        setCorrect((correct) => correct + 1);
        console.log(correct);
        return true;
      }
      Circle.tint = "#FFFF00";
    } else {
      setTries((tries) => tries + 1);
      while (stack.length !== 0) {
        const elem = stack.pop();
        resetColor(elem);
        elem.tint = "#FF0000";
        elem.interactive = false;
      }
    }
  }

  function resetColor(Graphics) {
    setTimeout(() => {
      Graphics.tint = undefined;
      Graphics.interactive = true;
    }, 800);
  }
  
  useEffect(() => {
    const gameIdfromStorage = localStorage.getItem('gameId');
    const childIdfromStorage = localStorage.getItem('childId');
    console.log("Child ID : ",childIdfromStorage);
    console.log("Game Id : ",gameIdfromStorage);
    if (gameIdfromStorage && childIdfromStorage){
      assignUploadUrl(`https://totalapi.joywithlearning.com/api/data/imagecapture/${gameIdfromStorage}/${childIdfromStorage}`);
    }
    assignInterval(5000); 
  }, []);

  useEffect(() => {
    if (!startTimeRef.current) {
      startTimeRef.current = new Date(); // Set start time only once
    }

    if (!gameOver) {
      startCapture();
      (async () => {
        // data.current = await (`../levels/level${lvl}/item${item}.json`);
        const response = await axios.get(`https://api.joywithlearning.com/api/connectingletters/${lvl}/${item}`);
        console.log(response);
        data.current = response.data;

        const app = new Application();
        appRef.current = app;

        let screenSize = {
          width: Math.min(window.innerHeight * 1.25, window.innerWidth * 0.9),
          height: Math.min(window.innerHeight * 0.8, window.innerWidth * 0.55)
        };
        await app.init({
          background: "#b7bce5",
          resolution: window.devicePixelRatio || 1, // Use device pixel ratio for better quality
          autoDensity: true,
          antialias: true,
          canvas: document.getElementById("board"),
          ...screenSize
        });

        app.renderer.resize(screenSize.width, screenSize.height);

        const texture = await Assets.load(
          (await import(`../levels/level${lvl}/images/item${item}.webp`)).default
        );

        let scalingFactor = 1;
        if (window.innerHeight < window.innerWidth)
          scalingFactor = (app.screen.height * 0.9) / texture.frame.height;
        else scalingFactor = (app.screen.width * 0.8) / texture.frame.width;

        const sprite = new Sprite(texture);
        sprite.scale = scalingFactor;
        const Padding = {
          x: (app.screen.width - sprite.width) / 2,
          y: (app.screen.height - sprite.height) / 2
        };
        sprite.x = Padding.x;
        sprite.y = Padding.y;
        app.stage.addChild(sprite);
        circles.current.forEach((circle) => {
          circle.off("pointerdown");
          app.stage.removeChild(circle);
        });
        circles.current = [];
        // await Assets.load('src/components/open_dyslexic/OpenDyslexic-Bold.otf');
        for (let i = 0; i < data.current.length; i++) {
          const word = Object.keys(data.current[i]);
          const pos = Object.values(data.current[i]);
          for (let j = 0; j < word.length; j++) {
            const px = pos[j][0] * scalingFactor + Padding.x;
            const py = pos[j][1] * scalingFactor + Padding.y;
            const letter = word[j];
            const LetterText = new Text({
              text: letter,
              style: new TextStyle({
                fontFamily: 'Arial',
                fontSize: 50 * scalingFactor,
                fill: "#000",
                align: "center"
              })
            });
            LetterText.x = px;
            LetterText.y = py;
            LetterText.anchor = 0.5;

            const Circle = new Graphics()
              .circle(px, py, 60 * scalingFactor)
              .fill({ color: "#fff" })
              .stroke({ color: "#000", width: 2 });
            Circle.word = i;
            Circle.letter = j;
            Circle.interactive = true;
            Circle.on("pointerdown", () => handleclick(Circle));
            app.stage.addChild(Circle);
            app.stage.addChild(LetterText);
          }
        }
      })();
    }
    // Cleanup on component unmount or when the game ends
    return () => {
      if(gameOver){
        endCapture();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lvl, item, gameOver]); // re-fetch data when level or item changes

  // Add a new useEffect to listen for URL changes and reload the page
  useEffect(() => {
    const handlePopState = () => {
      window.location.reload();
    };

    window.addEventListener('popstate', handlePopState);

    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  }, []);

  useEffect(() => {
    const popoverTriggerList = [].slice.call(
      document.querySelectorAll('[data-bs-toggle="popover"]')
    );
    const popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
      return new Popover(popoverTriggerEl);
    });
    return () => {
      popoverList.forEach((popover) => popover.dispose());
    };
  }, []);

  if (lvl !== 1 && lvl !== 2) return <h1 style={{ position: "absolute", top: '50%', left: '50%' }}>Not Found</h1>;
  return (
    <>
      <GameCompleted
        showModal={gameOver}
        setShowModal={setGameOver}
        handleRestart={handleRestart}
      />
      <div className="d-flex flex-column justify-content-center align-items-center pt-3">
        <div className="py-3 w-100">
          <div className="d-flex justify-content-around w-100">
            <div className="d-flex justify-content-start">
              <b className="fs-4" style={{ color: "green" }}>
                Correct {correct}
              </b>
            </div>
            <div className="d-flex justify-content-end">
              <b className="fs-4">
                L-{lvl} : I-{item}
              </b>
            </div>
          </div>
          <div className="d-flex justify-content-around w-100">
            <div className="d-flex justify-content-start">
              <b className="fs-4" style={{ color: "red" }}>
                Tries {tries}
              </b>
            </div>
            <div className="d-flex justify-content-end">
              <div
                className="align-items-center d-flex"
                data-bs-toggle="popover"
                data-bs-trigger="hover"
                data-bs-title="Instructions"
                data-bs-content="Click letters from left to right following the path"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  width={35}
                  height={35}
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="size-6"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z"
                  />
                </svg>
              </div>
              <button className="btn p-0 ms-2">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  width={35}
                  height={35}
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="size-6"
                  onClick={() => instruction.play()}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19.114 5.636a9 9 0 0 1 0 12.728M16.463 8.288a5.25 5.25 0 0 1 0 7.424M6.75 8.25l4.72-4.72a.75.75 0 0 1 1.28.53v15.88a.75.75 0 0 1-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.009 9.009 0 0 1 2.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75Z"
                  />
                </svg>
              </button>
            </div>
          </div>
        </div>
        <canvas id="board"></canvas>
      </div>
      <div>
        <NextLevel showModal={correct === 5 && !gameOver} nextLevel={handleNext} />
      </div>
      <CaptureElements/>
    </>
  );
}
