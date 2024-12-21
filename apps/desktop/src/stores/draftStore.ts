import { create } from "zustand";

interface DraftState {
  currentPatch: string;
  patches: string[];
  setCurrentPatch: (patch: string) => void;
  setPatchList: (patches: string[]) => void;
}

export const useDraftStore = create<DraftState>((set) => ({
  currentPatch: "",
  patches: [],
  setCurrentPatch: (patch) => set({ currentPatch: patch }),
  setPatchList: (patches) =>
    set({ patches, currentPatch: patches[patches.length - 1] }),
})); 