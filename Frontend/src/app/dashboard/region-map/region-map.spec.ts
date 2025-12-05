import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RegionMap } from './region-map';

describe('RegionMap', () => {
  let component: RegionMap;
  let fixture: ComponentFixture<RegionMap>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RegionMap]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RegionMap);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
