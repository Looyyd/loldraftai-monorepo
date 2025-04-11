import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@draftking/ui/components/ui/dialog";
import { InfoIcon } from "lucide-react";

export interface ChangelogModalProps {
  isOpen: boolean;
  closeHandler: () => void;
}

export const ChangelogModal: React.FC<ChangelogModalProps> = ({
  isOpen,
  closeHandler,
}) => {
  return (
    <Dialog open={isOpen} onOpenChange={closeHandler}>
      <DialogContent className="max-h-[80vh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <InfoIcon className="h-5 w-5" />
            Model Update
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm leading-relaxed">
              Model predictions for incomplete drafts, especially those with
              less than 5 champions picked, are now more accurate.
            </p>
            <p className="text-sm leading-relaxed">
              This should significantly improve champion suggestions for early
              draft picks.
            </p>
            <p className="text-sm leading-relaxed">
              Thanks to user Techniques on Discord for the report.
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};
