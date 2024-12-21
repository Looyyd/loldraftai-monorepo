import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import {
  CursorArrowRaysIcon,
  StarIcon,
  ChartBarIcon,
  PuzzlePieceIcon,
  ArrowsRightLeftIcon,
  TrashIcon,
} from "@heroicons/react/24/outline";

export interface HelpModalProps {
  isOpen: boolean;
  closeHandler: () => void;
}

export const HelpModal: React.FC<HelpModalProps> = ({ isOpen, closeHandler }) => {
  return (
    <Dialog open={isOpen} onOpenChange={closeHandler}>
      <DialogContent className="max-h-[80vh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>How to Use</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <p className="flex items-center">
            <CursorArrowRaysIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Click on a champion to add them to a team in pick order.
            </span>
          </p>
          <p className="flex items-center">
            <ArrowsRightLeftIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Select a team slot to swap champions or add a new one by clicking
              it.
            </span>
          </p>
          <p className="flex items-center">
            <StarIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Right-click (or long-press on mobile) on a champion to add them to
              your favorites.
            </span>
          </p>
          <p className="flex items-center">
            <TrashIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Right-click (or long-press on mobile) on a champion in team panels
              to remove them.
            </span>
          </p>
          <p className="flex items-center">
            <ChartBarIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Request a draft analysis to see win probabilities and suggestions.
            </span>
          </p>
          <p className="flex items-center">
            <PuzzlePieceIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Use the suggest champion button and favorites to get champion
              suggestions for a selected slots.
            </span>
          </p>
        </div>
        <p className="mt-6">
          Our machine learning model analyzes various features such as win
          rates, matchups, and team synergies to suggest the best champions for
          your draft.
        </p>
      </DialogContent>
    </Dialog>
  );
}; 