class ClusterManager:
    def __init__(self, reports_root: str):
        self.app = None
        self.base_path = reports_root
        self.context = None
        self.token_usage_p = 0
        self.token_usage_c = 0
        self.left_reports_txt = None
        self.left_reports_idx = None

    def initialize(self, app: int) -> None:
        self.app = app
        reports_path = f"{self.base_path}/app{self.app}.csv"
        reports = load_reports(reports_path)
        self.left_reports_txt = [r["description"] for r in reports]
        self.left_reports_idx = [i + 1 for i in range(len(reports))]

    def refresh(self) -> None:
        self.app = None
        self.context = None
        self.token_usage_p = 0
        self.token_usage_c = 0
        self.left_reports_txt = []
        self.left_reports_idx = []

    def run(self, app: int, limit: int = 50, show: bool = True) -> List[Tuple[int]]:
        self.refresh()
        self.initialize(app)
        self.iterative_clustering(limit)
        clusters = self.clusters_from_response()
        clusters = self.make_correction(clusters)
        self.generate_summaries()
        if show:
            for cluster in clusters:
                print(cluster)
        return clusters

    def iterative_clustering(self, limit: int) -> None:
        msg = None
        while len(self.left_reports_txt) > 0:
            print(len(self.left_reports_txt))

            max_num = min(limit, len(self.left_reports_txt))
            delta = len(self.left_reports_txt) - max_num
            if 0 < delta < 5:
                max_num = len(self.left_reports_txt)
            reports_txt = self.left_reports_txt[:max_num]
            reports_idx = self.left_reports_idx[:max_num]
            self.left_reports_txt = self.left_reports_txt[max_num:]
            self.left_reports_idx = self.left_reports_idx[max_num:]
            reports_str = "\n".join(f"{idx}: {txt}" for idx, txt in zip(reports_idx, reports_txt))

            if msg is None:
                prompt = cluster_sys_prompt()
                msg, pu, cu = query(system_msg=prompt, user_msg_txt=reports_str, model="gpt-4o")
            else:
                msg, pu, cu = query(context=msg, user_msg_txt=reports_str, model="gpt-4o")
            self.token_usage_p += pu
            self.token_usage_c += cu
        self.context = msg
        result = ""
        for m in msg:
            result += f"{m['role']}:\n{m['content']}\n\n"
        result += (f"Token Usage: {self.token_usage_p}"
                   f" + {self.token_usage_c} = {self.token_usage_p + self.token_usage_c}")
        with open(f"{self.base_path}/response/app{self.app}-cluster.txt", "w") as f:
            f.write(result)

    def clusters_from_response(self) -> List[Tuple[int]]:
        path = f"{self.base_path}/response/app{self.app}-cluster.txt"
        with open(path, mode="r", encoding="utf-8") as f:
            response = f.read()
        head = response.rfind("CURRENT CLUSTERS:") + len("CURRENT CLUSTERS:")
        tail = response.find("Token Usage:", head, len(response))
        tail = len(response) if tail == -1 else tail
        result = response[head:tail].strip()
        clusters = []
        start = None
        for i in range(len(result)):
            if result[i] == "{":
                start = i + 1
            elif result[i] == "}" and start is not None:
                try:
                    cluster = tuple(int(n.strip()) for n in result[start:i].split(","))
                except ValueError:
                    temp = []
                    for n in result[start:i].split(","):
                        j = n.find("-")
                        if j == -1:
                            temp.append(int(n.strip()))
                        else:
                            s, e = int(n[:j].strip()), int(n[j + 1:].strip())
                            for t in range(s, e + 1):
                                temp.append(t)
                    cluster = tuple(temp)
                clusters.append(cluster)
                start = None
        return clusters

    def make_correction(self, clusters_pred) -> List[Tuple]:
        reports_path = f"{self.base_path}/app{self.app}.csv"
        reports = load_reports(reports_path)
        items_true = [i + 1 for i in range(len(reports))]
        items_pred = []
        for cluster in clusters_pred:
            items_pred.extend(list(cluster))

        if len(items_pred) != len(set(items_pred)):
            diff_items = items_pred.copy()
            for item in set(items_pred):
                if item in diff_items:
                    diff_items.remove(item)
            for item in diff_items:
                related_clusters = []
                for cluster in clusters_pred:
                    if item in cluster:
                        related_clusters.append(cluster)
                related_clusters.sort(key=lambda x: len(x), reverse=True)
                for i in range(1, len(related_clusters)):
                    clusters_pred.remove(related_clusters[i])
                    new_cluster = tuple(set(related_clusters[i]) - {item})
                    if len(new_cluster) > 0:
                        clusters_pred.append(new_cluster)

        items_pred.sort()
        items_true.sort()
        omitted_items = tuple(set(items_true) - set(items_pred))
        if len(omitted_items) > 0:
            clusters_pred.append(omitted_items)
        return clusters_pred

    def generate_summaries(self):
        msg = self.context
        prompt = "Now summarize each cluster in one short phrase in English."
        msg, _, _ = query(context=msg, user_msg_txt=prompt, model="gpt-4o")
        result = msg[-1]["content"]
        with open(f"{self.base_path}/response/app{self.app}-summary.txt", "w") as f:
            f.write(result)

