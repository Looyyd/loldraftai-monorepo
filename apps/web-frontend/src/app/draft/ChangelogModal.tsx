import React from "react";
import Link from "next/link";
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
  version: string;
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
            Important Model Update
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm leading-relaxed">
              We&apos;ve fixed a bug in our model training process that was
              causing predictions to be overconfident. The new model should now
              provide:
            </p>
            <ul className="list-disc pl-5 space-y-1 text-sm">
              <li>More moderate win probability predictions</li>
              <li>More accurate overall assessments</li>
            </ul>
            <p className="text-sm text-muted-foreground mt-4">
              You might notice some changes in predictions compared to before.
              This is expected and represents more accurate assessments of draft
              outcomes.
            </p>
            <p className="text-sm mt-4">
              For more technical details about this update, you can read our{" "}
              <Link
                href="/blog/correction-reddit-post"
                className="text-primary hover:underline"
                onClick={closeHandler}
              >
                detailed blog post
              </Link>
              .
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};
