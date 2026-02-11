"use client";

import {
  CircleCheckIcon,
  InfoIcon,
  Loader2Icon,
  OctagonXIcon,
  TriangleAlertIcon,
} from "lucide-react";
import { useTheme } from "next-themes";
import { Toaster as Sonner, type ToasterProps } from "sonner";

const Toaster = ({ toastOptions, ...props }: ToasterProps) => {
  const { theme = "system" } = useTheme();
  const mergedToastOptions: ToasterProps["toastOptions"] = {
    duration: 4600,
    ...toastOptions,
    classNames: {
      toast:
        "group rounded-xl border border-border/70 bg-card/95 text-foreground shadow-xl backdrop-blur-md",
      title: "text-sm font-semibold tracking-tight",
      description: "text-xs text-muted-foreground",
      actionButton:
        "rounded-md border border-primary/30 bg-primary text-primary-foreground",
      cancelButton: "rounded-md border border-border bg-muted text-foreground",
      closeButton:
        "rounded-md border border-border/70 bg-background/80 text-muted-foreground hover:text-foreground",
      ...toastOptions?.classNames,
    },
  };

  return (
    <Sonner
      theme={theme as ToasterProps["theme"]}
      className="toaster group"
      position="bottom-right"
      closeButton
      expand={false}
      visibleToasts={4}
      offset={18}
      icons={{
        success: <CircleCheckIcon className="size-4" />,
        info: <InfoIcon className="size-4" />,
        warning: <TriangleAlertIcon className="size-4" />,
        error: <OctagonXIcon className="size-4" />,
        loading: <Loader2Icon className="size-4 animate-spin" />,
      }}
      toastOptions={mergedToastOptions}
      style={
        {
          "--normal-bg": "var(--popover)",
          "--normal-text": "var(--popover-foreground)",
          "--normal-border": "var(--border)",
          "--border-radius": "var(--radius)",
        } as React.CSSProperties
      }
      {...props}
    />
  );
};

export { Toaster };
