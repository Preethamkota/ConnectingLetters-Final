import { Modal, Button } from "react-bootstrap";

export default function GameCompleted({ showModal, setShowModal, handleRestart }) {
  return (
    <Modal centered show={showModal}>
      <Modal.Header>
        <Modal.Title className="mx-auto">Game Over</Modal.Title>
      </Modal.Header>
      <Modal.Body>Congratulations! You've completed the game.</Modal.Body>
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
