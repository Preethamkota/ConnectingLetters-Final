import { Modal, Button } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css"

export default function NextLevel({ showModal, nextLevel }) {
  return (
    <Modal centered show={showModal}>
      <Modal.Header>
        <Modal.Title className="mx-auto">Stage Completed</Modal.Title>
      </Modal.Header>
      <Modal.Body className="mx-auto">
        <Button variant="secondary" className="py-2 px-3" onClick={() => nextLevel(false)}>
          Next
        </Button>
      </Modal.Body>
    </Modal>
  );
}
