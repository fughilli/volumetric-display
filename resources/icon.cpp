#include "resources/icon.h"

#include "resources/icon_helper.h"

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "tools/cpp/runfiles/runfiles.h"

using bazel::tools::cpp::runfiles::Runfiles;

void SetIcon(std::string argv0) {
  std::string error;
  static std::unique_ptr<Runfiles> runfiles(Runfiles::Create(argv0, &error));

  if (runfiles == nullptr) {
    LOG(ERROR) << "Failed to create Runfiles: " << error;
    return;
  }

  std::string icon_path = runfiles->Rlocation("volumetric-display/resources/icon.png");

  if (icon_path.empty()) {
    LOG(ERROR) << "Failed to find icon.png";
    return;
  }

  SetIconHelper(icon_path);
}
