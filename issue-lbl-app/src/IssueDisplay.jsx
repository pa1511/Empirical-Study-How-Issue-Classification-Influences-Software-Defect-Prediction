import { useCallback } from "react";
import Glyphicon from "@strongdm/glyphicon";
import { Alert, Button, Col, Row } from "react-bootstrap";
//
const TYPE_MAP = {
	Feature: "success",
	Modification: "warning",
	Bug: "danger",
	Other: "secondary",
};
function IssueDisplay(props) {
	const setFeature = useCallback(() => props.onSetIssueType("Feature"), [props]);
	const setModification = useCallback(() => props.onSetIssueType("Modification"), [props]);
	const setBug = useCallback(() => props.onSetIssueType("Bug"), [props]);
	const setOther = useCallback(() => props.onSetIssueType("Other"), [props]);
	//
	const checkAndSetAlert = useCallback((type) => props.issue["type"] === type && <Alert variant={TYPE_MAP[type]}>{type}</Alert>, [props]);
	//
	if (props.issue === undefined) {
		return <>TODO: issue has become undefined. This should not happen!</>;
	}
	//
	return (
		<>
			<Row>
				{props.issue && (
					<Col md={8}>
						<Row>
							<Col>
								<h4>{`${props.issue["title"]} ==> ${props.issue["number"]}`}</h4>
							</Col>
						</Row>
						<Row>
							<Col md={1}>
								PR: <i>{props.issue["has_pull_request"] === true ? "Y" : "N"}</i>
							</Col>
							<Col md={1}>
								WM: <i>{props.issue["pull_request_was_merged"] === true ? "Y" : "N"}</i>
							</Col>
							<Col md={10}>
								{props.issue["labels"].map((tag, lbl_id) => {
									var backgroundColor = "#FDFD96";
									if (tag.toLowerCase().includes("feature")) {
										backgroundColor = "#A7E99C";
									} else if (tag.toLowerCase().includes("bug")) {
										backgroundColor = "#FF6961";
									}
									//
									return (
										<span
											key={lbl_id}
											style={{
												fontSize: "x-small",
												background: backgroundColor,
												margin: "3px",
												padding: "4px",
												borderRadius: "10px",
												display: "inline-block",
											}}
										>
											{tag}
										</span>
									);
								})}
							</Col>
						</Row>
					</Col>
				)}
				<Col md={4}>
					<Row>
						<Col md={2}>
							<Button onClick={props.onRequestPreviousIssue}>
								<Glyphicon glyph="chevron-left" />
							</Button>
						</Col>
						<Col md={6}>{props.displayCurrentIssue()}</Col>
						<Col md={2}>
							<Button onClick={props.onRequestNextIssue}>
								<Glyphicon glyph="chevron-right" />
							</Button>
						</Col>
						<Col md={2}>
							<Button variant="danger" onClick={props.onRequestDelete}>
								<Glyphicon glyph="trash" />
							</Button>
						</Col>
					</Row>
				</Col>
			</Row>
			{props.issue && (
				<>
					<hr style={{ background: "none", borderTop: "1px dashed black" }} />
					<Row>
						<Col md={8}>
							{checkAndSetAlert("Feature")}
							{checkAndSetAlert("Modification")}
							{checkAndSetAlert("Bug")}
							{checkAndSetAlert("Other")}
						</Col>
						<Col md={4}>
							<div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "5px" }}>
								<Button onClick={setFeature} variant={TYPE_MAP["Feature"]}>
									F
								</Button>
								<Button onClick={setModification} variant={TYPE_MAP["Modification"]}>
									M
								</Button>
								<Button onClick={setBug} variant={TYPE_MAP["Bug"]}>
									B
								</Button>
								<Button onClick={setOther} variant={TYPE_MAP["Other"]}>
									O
								</Button>
							</div>
						</Col>
					</Row>
					<hr style={{ background: "none", borderTop: "1px dashed black" }} />
					<Row>
						<Col>
							<b>Description:</b>
						</Col>
					</Row>
					<Row>
						<Col>
							<pre style={{ width: "100%", maxHeight: "55vh", overflow: "scroll" }}>{props.issue["body"]}</pre>
						</Col>
					</Row>
				</>
			)}
		</>
	);
}

export default IssueDisplay;
