import Glyphicon from "@strongdm/glyphicon";
import { useState } from "react";
import { Button, Col, Row } from "react-bootstrap";

function RepoDisplay(props) {
	const [saveFileName, setSaveFileName] = useState("");

	return (
		<Row>
			<Col md={4}>
				<h2>{props.repo}</h2>
			</Col>
			<Col md={4}>
				<label>Save file:</label>
				<input type="text" value={saveFileName} onChange={e=>setSaveFileName(e.target.value)} />
				<Button onClick={() => props.onDownload(saveFileName)}>
					<Glyphicon glyph="download-alt" />
				</Button>
			</Col>
			<Col md={4}>
				<Row>
					<Col md={2}>
						<Button onClick={props.onRequestPreviousRepo}>
							<Glyphicon glyph="chevron-left" />
						</Button>
					</Col>
					<Col md={6}>{props.displayCurrentRepo()}</Col>
					<Col md={2}>
						<Button onClick={props.onRequestNextRepo}>
							<Glyphicon glyph="chevron-right" />
						</Button>
					</Col>
				</Row>
			</Col>
		</Row>
	);
}

export default RepoDisplay;
