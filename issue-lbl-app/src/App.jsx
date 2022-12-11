import { useCallback, useEffect, useState } from "react";
import { Container } from "react-bootstrap";
import IssueDisplay from "./IssueDisplay";
//
import RepoDisplay from "./RepoDisplay";

function App() {
	const [data, setData] = useState();
	const [repoId, setRepoId] = useState(-1);
	const [issueId, setIssueId] = useState(-1);
	//
	useEffect(() => {
		if (data !== undefined) {
			if (repoId !== -1) {
				const issue = parseInt(localStorage.getItem(`last_issue_of_${repoId}`)) || 0;
				if (issue < data[repoId].length) {
					setIssueId(issue);
				}
				//
				localStorage.setItem("last_repo", repoId);
			}
		}
	}, [data, repoId]);
	//
	useEffect(() => {
		if (data !== undefined) {
			if (issueId !== -1) {
				localStorage.setItem(`last_issue_of_${repoId}`, issueId);
			}
		}
	}, [data, repoId, issueId]);
	//
	const download = useCallback((fileName) => {
		var a = document.createElement("a");
		var file = new Blob([JSON.stringify(data)], { type: "text/json" });
		a.href = URL.createObjectURL(file);
		a.download = fileName ?? "labeled_issue_lvl.json";
		a.click();
	}, [data]);
	//
	const deleteIssue = useCallback(() => {
		setData((data) => {
			const newData = {...data};
			newData[repoId].splice(issueId, 1);
			setIssueId((id) => {
				return Math.min(id, newData[repoId].length - 1);
			});
			return newData;
		});
	}, [repoId, issueId]);
	//
	const onRequestPreviousRepo = useCallback(() => {
		const repos = Object.keys(data);
		repos.sort();
		const currentId = repos.indexOf(repoId);
		setRepoId(repos[Math.max(currentId - 1, 0)]);
	}, [data, repoId]);
	const onRequestNextRepo = useCallback(() => {
		const repos = Object.keys(data);
		repos.sort();
		const currentId = repos.indexOf(repoId);
		setRepoId(repos[Math.min(currentId + 1, repos.length - 1)]);
	}, [data, repoId]);
	//
	const onRequestPreviousIssue = useCallback(() => setIssueId((id) => Math.max(id - 1, 0)), []);
	const onRequestNextIssue = useCallback(() => setIssueId((id) => Math.min(id + 1, data[repoId].length - 1)), [data, repoId]);
	const goToIssue = useCallback(
		(e) => {
			var newIssue = 0;
			if (e.target.value !== "" && e.target.value !== "1") {
				newIssue = parseInt(e.target.value) - 1 || issueId;
				newIssue = Math.min(newIssue, data[repoId].length - 1);
				newIssue = Math.max(newIssue, 0);
			}
			setIssueId(newIssue);
		},
		[data, repoId, issueId]
	);
	//
	const setType = useCallback(
		(type) =>
			setData((data) => {
				const newData = {...data};
				newData[repoId][issueId]["type"] = type;
				return newData;
			}),
		[repoId, issueId]
	);
	//
	return (
		<>
			<br />
			<Container>
				{(data !== undefined && (
					<>
						<RepoDisplay
							repo={repoId}
							displayCurrentRepo={() => (
								<p>{repoId}</p>
							)}
							onRequestPreviousRepo={onRequestPreviousRepo}
							onRequestNextRepo={onRequestNextRepo}
							onDownload={download}
						/>
						<hr />
						<IssueDisplay
							issue={data[repoId][issueId]}
							displayCurrentIssue={() => (
								<span>
									<input
										onChange={goToIssue}
										style={{ display: "inline-block", width: "70%" }}
										pattern="[0-9]*"
										type="text"
										value={issueId + 1}
									/>
									{`/${data[repoId].length}`}
								</span>
							)}
							onRequestPreviousIssue={onRequestPreviousIssue}
							onRequestNextIssue={onRequestNextIssue}
							onRequestDelete={deleteIssue}
							onSetIssueType={setType}
						/>
					</>
				)) || (
					<>
						<h4>Please select your data file:</h4>
						<input
							type="file"
							onChange={(e) => {
								const file = e.target.files[0];
								file.text().then((encodedData) => {
									const data = JSON.parse(encodedData);
									const repos = Object.keys(data);
									repos.sort();
									if(repos.length>0 && Object.prototype.toString.call(data[repos[0]]) !== "[object Array]"){
										repos.forEach(repo => {
											data[repo] = Object.values(data[repo]);
										});	
									}
									//
									const startRepo = parseInt(localStorage.getItem("last_repo")) || repos[0];
									setRepoId(startRepo);
									//
									const startIssue = parseInt(localStorage.getItem(`last_issue_of_${startRepo}`)) || 0;
									if (startIssue < data[startRepo].length) {
										setIssueId(startIssue);
									}
									//
									setData(data);
								});
							}}
							accept=".json"
						/>
					</>
				)}
			</Container>
		</>
	);
}

export default App;
