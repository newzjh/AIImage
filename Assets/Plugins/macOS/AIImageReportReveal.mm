#import <Cocoa/Cocoa.h>

extern "C"
{
    int AIImageMacReportReveal(const char* path)
    {
        NSString* value = path == nullptr ? @"" : [NSString stringWithUTF8String:path];
        if (value.length == 0)
            return 0;

        if (![[NSFileManager defaultManager] fileExistsAtPath:value])
        {
            NSLog(@"[Aexis] Runner report does not exist: %@", value);
            return 0;
        }

        __block BOOL selected = NO;
        void (^reveal)(void) = ^{
            NSWorkspace* workspace = [NSWorkspace sharedWorkspace];
            selected = [workspace selectFile:value inFileViewerRootedAtPath:@""];
            if (!selected)
            {
                NSURL* url = [NSURL fileURLWithPath:value];
                [workspace activateFileViewerSelectingURLs:@[url]];
            }
        };
        if ([NSThread isMainThread])
            reveal();
        else
            dispatch_sync(dispatch_get_main_queue(), reveal);
        return selected ? 1 : 0;
    }
}
