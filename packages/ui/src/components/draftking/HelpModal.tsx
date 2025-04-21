import React from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";
import {
  CursorArrowRaysIcon,
  StarIcon,
  ArrowsRightLeftIcon,
  TrashIcon,
} from "@heroicons/react/24/outline";
import { SparklesIcon, LightBulbIcon } from "@heroicons/react/24/solid";

export interface HelpModalProps {
  isOpen: boolean;
  closeHandler: () => void;
}

export const HelpModal: React.FC<HelpModalProps> = ({
  isOpen,
  closeHandler,
}) => {
  return (
    <Dialog open={isOpen} onOpenChange={closeHandler}>
      <DialogContent className="max-h-[80vh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>
            How to use <span className="brand-text">LoLDraftAI</span>
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <p className="flex items-center">
            <CursorArrowRaysIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Click on a champion to add them to a team. Teams will be
              automatically reassigned to the most likely champion/role
              combination. Use the lock button to prevent automatic
              reassignments
            </span>
          </p>
          <p className="flex items-center">
            <ArrowsRightLeftIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Click on a team position to swap champions or add a new champion
              to the position
            </span>
          </p>
          <p className="flex items-center">
            <TrashIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Click the bin icon or right-click on a champion in team panels to
              remove them from the team
            </span>
          </p>
          <p className="flex items-center">
            <SparklesIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Request a draft analysis to see win probabilities as well as
              additional predictions
            </span>
          </p>
          <p className="flex items-center">
            <StarIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Right-click on a champion in the grid to add them to your
              favorites
            </span>
          </p>
          <p className="flex items-center">
            <LightBulbIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Use the "Suggest Champion" button to get suggestions for a
              selected position, suggestions can be filtered by favorites only
            </span>
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
};
