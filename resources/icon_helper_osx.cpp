#include "resources/icon_helper.h"

#include <Cocoa/Cocoa.h>

#include <string>

#include "absl/log/log.h"

void SetIconHelper(std::string icon_path) {
  @autoreleasepool {
    NSImage *icon = [[NSImage alloc]
        initWithContentsOfFile:[NSString
                                   stringWithUTF8String:icon_path.c_str()]];
    if (icon == nil) {
      LOG(ERROR) << "Failed to load icon.png";
      return;
    }
    [NSApp setApplicationIconImage:icon];
  }
}
