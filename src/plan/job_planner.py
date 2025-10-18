from typing import List
from logzero import logger
from src.constant import CONSTANT
from src.floor import Coordinate, SectorMapSnapshot
from src.job import InstructionType, Job, JobInstruction
from src.operators import HT_Coordinate_View
from src.plan.job_tracker import JobTracker


class JobPlanner:
    """
    Coordinates job planning activities using HT tracker and sector map data.
    """

    def __init__(
        self,
        ht_coord_tracker: HT_Coordinate_View,
        sector_map_snapshot: SectorMapSnapshot,
    ):
        self.ht_coord_tracker = ht_coord_tracker
        self.sector_map_snapshot = sector_map_snapshot

    def is_deadlock(self):
        return self.ht_coord_tracker.is_deadlock()

    def get_non_moving_HT(self):
        return self.ht_coord_tracker.get_non_moving_HT()

    # ===================================================
    # UPDATED PLAN() WITH IMPROVED CALLS FOR FLEXIBLE USE
    # ===================================================
    def plan(self, job_tracker: JobTracker) -> List[Job]:
        plannable_job_seqs = job_tracker.get_plannable_job_sequences()
        selected_HT_names = []
        new_jobs = []

        for job_seq in plannable_job_seqs:
            job = job_tracker.get_job(job_seq)
            job_info = job.get_job_info()
            job_type, QC_name, yard_name, alt_yard_names = [
                job_info[k]
                for k in ["job_type", "QC_name", "yard_name", "alt_yard_names"]
            ]

            HT_name = self.select_HT(
                job_type=job_type,
                QC_name=QC_name,
                yard_name=yard_name,
                selected_HT_names=selected_HT_names,
            )

            if HT_name is None:
                break
            selected_HT_names.append(HT_name)

            yard_name = self.select_yard(
                preferred_yard=yard_name,
                alt_yard_names=alt_yard_names or [],
                job_type=job_type,
                QC_name=QC_name,
            )

            job.assign_job(HT_name=HT_name, yard_name=yard_name)
            job_instructions = []
            buffer_coord = self.ht_coord_tracker.get_coordinate(HT_name)

            if job_type == CONSTANT.JOB_PARAMETER.DISCHARGE_JOB_TYPE:
                # 1) QC
                job_instructions.append(JobInstruction(instruction_type=InstructionType.BOOK_QC))
                path = self.get_path_from_buffer_to_QC(buffer_coord, QC_name)
                job_instructions.append(JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path))
                job_instructions.append(JobInstruction(instruction_type=InstructionType.WORK_QC, HT_name=HT_name, QC_name=QC_name))
                path = self.get_path_from_QC_to_buffer(QC_name, buffer_coord)
                job_instructions.append(JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path))

                # 2) Yard
                job_instructions.append(JobInstruction(instruction_type=InstructionType.BOOK_YARD))
                path = self.get_path_from_buffer_to_yard(buffer_coord, yard_name)
                job_instructions.append(JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path))
                job_instructions.append(JobInstruction(instruction_type=InstructionType.WORK_YARD, HT_name=HT_name, yard_name=yard_name))
                path = self.get_path_from_yard_to_buffer(yard_name, buffer_coord)
                job_instructions.append(JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path))

            else:  # LO job
                job_instructions.append(JobInstruction(instruction_type=InstructionType.BOOK_YARD))
                path = self.get_path_from_buffer_to_yard(buffer_coord, yard_name)
                job_instructions.append(JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path))
                job_instructions.append(JobInstruction(instruction_type=InstructionType.WORK_YARD, HT_name=HT_name, yard_name=yard_name))
                path = self.get_path_from_yard_to_buffer(yard_name, buffer_coord)
                job_instructions.append(JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path))

                job_instructions.append(JobInstruction(instruction_type=InstructionType.BOOK_QC))
                path = self.get_path_from_buffer_to_QC(buffer_coord, QC_name)
                job_instructions.append(JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path))
                job_instructions.append(JobInstruction(instruction_type=InstructionType.WORK_QC, HT_name=HT_name, QC_name=QC_name))
                path = self.get_path_from_QC_to_buffer(QC_name, buffer_coord)
                job_instructions.append(JobInstruction(instruction_type=InstructionType.DRIVE, HT_name=HT_name, path=path))

            job.set_instructions(job_instructions)
            new_jobs.append(job)

        return new_jobs

    # ===================================================
    # SMARTER HELPER UTILITIES
    # ===================================================
    def _get_bounds(self):
        try:
            W = getattr(self.sector_map_snapshot, "width", 42)
            H = getattr(self.sector_map_snapshot, "height", 12)
            return 1, int(W), 1, int(H)
        except Exception:
            return 1, 42, 1, 12

    def _manhattan(self, a: Coordinate, b: Coordinate) -> int:
        return abs(a.x - b.x) + abs(a.y - b.y)

    def _get_buffer_coord(self, HT_name: str) -> Coordinate:
        return self.ht_coord_tracker.get_coordinate(HT_name)

    # ===================================================
    # AI-INSPIRED HT SELECTION (HYBRID SCORING)
    # ===================================================
    def select_HT(self, job_type: str, QC_name: str, yard_name: str, selected_HT_names: List[str]) -> str | None:
        available_HTs = [ht for ht in self.ht_coord_tracker.get_available_HTs() if ht not in selected_HT_names]
        if not available_HTs:
            return None

        if job_type == CONSTANT.JOB_PARAMETER.DISCHARGE_JOB_TYPE:
            start = self.sector_map_snapshot.get_QC_sector(QC_name).in_coord
        else:
            start = self.sector_map_snapshot.get_yard_sector(yard_name).in_coord

        ALPHA_DIST, BETA_FREE = 1.0, 0.2

        def time_until_free(ht):
            fn = getattr(self.ht_coord_tracker, "get_time_until_free", None)
            if callable(fn):
                try:
                    return fn(ht)
                except Exception:
                    return 0
            return 0

        best, best_score = None, float("inf")
        for ht in available_HTs:
            coord = self._get_buffer_coord(ht)
            dist = self._manhattan(coord, start)
            free = time_until_free(ht)
            score = ALPHA_DIST * dist + BETA_FREE * free
            if score < best_score:
                best, best_score = ht, score

        return best

    # ===================================================
    # SMART YARD CHOICE (CAPACITY-AWARE)
    # ===================================================
    def select_yard(self, preferred_yard: str, alt_yard_names: List[str], job_type: str, QC_name: str) -> str:
        seen, candidates = set(), []
        for y in [preferred_yard, *alt_yard_names]:
            if y and y not in seen:
                candidates.append(y)
                seen.add(y)
        if not candidates:
            return preferred_yard

        try:
            ref = (
                self.sector_map_snapshot.get_QC_sector(QC_name).in_coord
                if job_type == CONSTANT.JOB_PARAMETER.DISCHARGE_JOB_TYPE
                else Coordinate(21, 7)
            )
        except Exception:
            ref = Coordinate(21, 7)

        def yard_ok_and_cost(y):
            try:
                ysec = self.sector_map_snapshot.get_yard_sector(y)
                count = int(getattr(ysec, "current_di_job_count", 0))
                if count >= 700:
                    return (False, float("inf"))
                dist = self._manhattan(ref, ysec.in_coord)
                return (True, dist)
            except Exception:
                return (True, 10_000)

        best, best_cost = None, float("inf")
        for y in candidates:
            ok, cost = yard_ok_and_cost(y)
            if ok and cost < best_cost:
                best, best_cost = y, cost
        return best if best else preferred_yard

    # ===================================================
    # PATH HELPERS (OPTIMIZED)
    # ===================================================
    def _line_path(self, start: Coordinate, end: Coordinate):
        path = []
        x_step = 1 if end.x > start.x else -1
        for x in range(start.x, end.x, x_step):
            path.append(Coordinate(x, start.y))
        y_step = 1 if end.y > start.y else -1
        for y in range(start.y, end.y, y_step):
            path.append(Coordinate(end.x, y))
        path.append(end)
        return path

    def _stitch(self, segs):
        out = []
        for s in segs:
            for c in s:
                if not out or (out[-1].x != c.x or out[-1].y != c.y):
                    out.append(c)
        return out

    QC_TRAVEL_Y = 4
    QC_APPROACH_Y = 5
    HWY_TOP_Y = 7
    HWY_BOT2_Y = 12

    def get_path_from_buffer_to_QC(self, buffer_coord: Coordinate, QC_name: str) -> List[Coordinate]:
        qc_in = self.sector_map_snapshot.get_QC_sector(QC_name).in_coord
        if buffer_coord.x <= qc_in.x:
            seg1 = self._line_path(buffer_coord, Coordinate(buffer_coord.x, self.QC_TRAVEL_Y))
            seg2 = self._line_path(seg1[-1], Coordinate(qc_in.x, self.QC_TRAVEL_Y))
            seg3 = self._line_path(seg2[-1], qc_in)
            return self._stitch([seg1, seg2, seg3])
        xmin, xmax, _, _ = self._get_bounds()
        s1 = self._line_path(buffer_coord, Coordinate(buffer_coord.x, self.HWY_BOT2_Y))
        s2 = self._line_path(s1[-1], Coordinate(xmax, self.HWY_BOT2_Y))
        s3 = self._line_path(s2[-1], Coordinate(qc_in.x, self.QC_TRAVEL_Y))
        s4 = self._line_path(s3[-1], qc_in)
        return self._stitch([s1, s2, s3, s4])

    def get_path_from_QC_to_buffer(self, QC_name: str, buffer_coord: Coordinate) -> List[Coordinate]:
        qc_out = self.sector_map_snapshot.get_QC_sector(QC_name).out_coord
        xmin, xmax, _, _ = self._get_bounds()
        s1 = self._line_path(qc_out, Coordinate(qc_out.x, self.QC_TRAVEL_Y))
        s2 = self._line_path(s1[-1], Coordinate(xmax - 1, self.QC_TRAVEL_Y))
        s3 = self._line_path(s2[-1], Coordinate(xmax - 1, self.HWY_TOP_Y))
        s4 = self._line_path(s3[-1], Coordinate(buffer_coord.x, self.HWY_TOP_Y))
        s5 = self._line_path(s4[-1], buffer_coord)
        return self._stitch([s1, s2, s3, s4, s5])

    def get_path_from_buffer_to_yard(self, buffer_coord: Coordinate, yard_name: str) -> List[Coordinate]:
        yard_in = self.sector_map_snapshot.get_yard_sector(yard_name).in_coord
        xmin, xmax, _, _ = self._get_bounds()
        s1 = self._line_path(buffer_coord, Coordinate(buffer_coord.x, self.QC_APPROACH_Y))
        s2 = self._line_path(s1[-1], Coordinate(min(yard_in.x, xmax), self.QC_APPROACH_Y))
        s3 = self._line_path(s2[-1], Coordinate(yard_in.x, self.HWY_BOT2_Y))
        s4 = self._line_path(s3[-1], yard_in)
        return self._stitch([s1, s2, s3, s4])

    def get_path_from_yard_to_buffer(self, yard_name: str, buffer_coord: Coordinate) -> List[Coordinate]:
        y_out = self.sector_map_snapshot.get_yard_sector(yard_name).out_coord
        xmin, xmax, _, _ = self._get_bounds()
        s1 = self._line_path(y_out, Coordinate(y_out.x, self.HWY_BOT2_Y))
        s2 = self._line_path(s1[-1], Coordinate(xmax - 1, self.HWY_BOT2_Y))
        s3 = self._line_path(s2[-1], Coordinate(xmax - 1, self.HWY_TOP_Y))
        s4 = self._line_path(s3[-1], Coordinate(buffer_coord.x, self.HWY_TOP_Y))
        s5 = self._line_path(s4[-1], buffer_coord)
        return self._stitch([s1, s2, s3, s4, s5])
