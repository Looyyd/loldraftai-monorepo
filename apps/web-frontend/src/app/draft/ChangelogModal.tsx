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
            UI Update
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm leading-relaxed">
              When adding a champion to a team, all champions will now be
              reassigned to get the most likely team combination. To avoid
              reassignment on a champion, use the lock button.
            </p>
            <p className="text-sm leading-relaxed">
              This also works when live tracking in the desktop app.
            </p>
            <p className="text-sm leading-relaxed">
              Example: You have a Cho&apos;Gath Top, but now you add Renekton to
              the team. The most likely combination is now Cho&apos;Gath Mid and
              Renekton Top, so champions will be assigned to these roles.
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};
