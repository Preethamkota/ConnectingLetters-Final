import { Routes, Route } from "react-router-dom";
import "./index.css";
import Game from "./components/Game";

import l1 from "./assets/L1img.jpg";
import l2 from "./assets/L2img.jpg";
import { Link } from "react-router-dom";

export default function App() {
  return (
    <>
      <Routes>
        <Route path="/game" element={<Game />} />
        <Route
          path="/"
          element={
            <section className="container">
              <div className="d-flex flex-column align-items-center">
                <h1 className="p-5">Connecting Letters</h1>
                <div className="d-flex flex-row justify-content-around align-items-center flex-wrap ">
                  <div
                    className="card  border border-4 m-3"
                    style={{ maxWidth: "320px", minWidth: "280px" }}
                  >
                    <Link
                      className="card-body d-flex flex-column flex-wrap justify-content-center text-decoration-none"
                      to="./game?lvl=1"
                    >
                      <h4 className="card-title">Level 1</h4>
                      <img
                        src={l1}
                        alt="Display"
                        className="img-fluid rounded-3 border border-2 border-secondary my-2"
                      />
                      <p className="card-text">
                        Connect a letter on left side of the page to a letter on right side of the
                        page by following the line.
                      </p>
                    </Link>
                  </div>

                  <div
                    className="card  border border-4 mx-3"
                    style={{ maxWidth: "320px", minWidth: "280px" }}
                  >
                    <Link
                      className="card-body d-flex flex-column flex-wrap justify-content-center text-decoration-none"
                      to="./game?lvl=2"
                    >
                      <h4 className="card-title">Level 2</h4>
                      <img
                        src={l2}
                        alt="Display"
                        className="img-fluid rounded-3 border border-2 border-secondary my-2"
                      />
                      <p className="card-text">
                        Connect all letters along the string from left side of the page to the right
                        to form words.
                      </p>
                    </Link>
                  </div>
                </div>
              </div>
            </section>
          }
        />
      </Routes>
    </>
  );
}
