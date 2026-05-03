import { Modal, Button } from "react-bootstrap";

export default function GameCompleted({
  showModal,
  setShowModal,
  handleRestart,
  metrics,
  analysisSummary,
}) {
  const emotionPercentages = analysisSummary?.emotion?.percentages || {};
  const gazeAnalysis = analysisSummary?.gaze || {};

  return (
    <Modal centered show={showModal}>
      <Modal.Header>
        <Modal.Title className="mx-auto">Game Over</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <p>Congratulations! You've completed the game.</p>
        <p className="mb-1">Accuracy: {metrics.accuracy.toFixed(1)}%</p>
        <p className="mb-1">Total Attempts: {metrics.totalAttempts}</p>
        <p className="mb-1">
          Correct / Incorrect: {metrics.correctCount} / {metrics.incorrectCount}
        </p>
        <p className="mb-1">Average Response Time: {metrics.averageResponseTime.toFixed(2)}s</p>
        <p className="mb-0">Error Count: {metrics.errorCount}</p>
        <hr />
        <p className="mb-1">Focused Time: {(gazeAnalysis.percentTimeFocused || 0).toFixed(1)}%</p>
        <p className="mb-1">Distractions: {gazeAnalysis.distractions || 0}</p>
        <p className="mb-1">
          Average Focus Duration: {(gazeAnalysis.averageFocusDuration || 0).toFixed(2)}s
        </p>
        <p className="mb-1">Happy: {(emotionPercentages.happy || 0).toFixed(1)}%</p>
        <p className="mb-1">Neutral: {(emotionPercentages.neutral || 0).toFixed(1)}%</p>
        <p className="mb-1">Frustrated: {(emotionPercentages.frustrated || 0).toFixed(1)}%</p>
        <p className="mb-0">Confused: {(emotionPercentages.confused || 0).toFixed(1)}%</p>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={() => setShowModal(false)}>
          Close
        </Button>
        <Button variant="primary" onClick={handleRestart}>
          Restart Game
        </Button>
      </Modal.Footer>
    </Modal>
  );
}
